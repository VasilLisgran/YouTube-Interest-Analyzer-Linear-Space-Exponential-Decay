import argparse
import json
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from pipeline_common import (STOP_WORDS, category_document_frequency,
                             keyword_score, load_events)

warnings.filterwarnings("ignore")

def load_user(path, dedup_key='video_id'):
    return load_events(path, dedup_key=dedup_key, require_date=False)

def stratified_split(videos, test_frac, seed):
    rng = np.random.default_rng(seed)
    by_cat = defaultdict(list)
    for v in videos:
        by_cat[v['category']].append(v)
    train, test = [], []
    for cat, items in sorted(by_cat.items()):
        idx = rng.permutation(len(items))
        n_test = int(round(len(items) * test_frac))
        if len(items) < 2:
            n_test = 0
        n_test = min(n_test, len(items) - 1)
        for j, i in enumerate(idx):
            (test if j < n_test else train).append(items[i])
    return train, test

_MODEL_CACHE = {}

class Embedder:
    """fit() on train texts only; transform() applies the fitted space.

    LaBSE is a frozen pretrained embedder -- it is never fit on this dataset,
    so sharing one loaded instance across every account and every seed changes
    nothing about the results; it only avoids reloading the model weights on
    every split.  An earlier version created a new Embedder() per seed, so the
    model was reloaded ~20 times per account; on Apple Silicon each reload
    leaves MPS allocator state behind that is not always reclaimed, and the
    run eventually failed with "MPS backend out of memory".  Loading once into
    a module-level cache removes both the slowdown and the crash.  Device is
    pinned to CPU: LaBSE is small enough that CPU inference on a few hundred
    short titles per split is fast, and it sidesteps the MPS allocator
    entirely rather than trying to manage it correctly.
    """

    def __init__(self, kind):
        self.kind = kind
        self.model = None
        self.vec = None

    def fit(self, texts):
        if self.kind == 'labse':
            if 'labse' not in _MODEL_CACHE:
                from sentence_transformers import SentenceTransformer
                print('  [loading LaBSE once, reused for all users/seeds]')
                _MODEL_CACHE['labse'] = SentenceTransformer(
                    'sentence-transformers/LaBSE', device='cpu')
            self.model = _MODEL_CACHE['labse']
        elif self.kind == 'tfidf':
            self.vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                       min_df=1, sublinear_tf=True).fit(texts)
        elif self.kind == 'char':
            self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                       min_df=2, sublinear_tf=True).fit(texts)
        return self

    def transform(self, texts):
        if self.kind == 'labse':
            X = self.model.encode(texts, show_progress_bar=False)
        else:
            X = self.vec.transform(texts).toarray()
        return normalize(np.asarray(X, dtype=float))

# proposed method: per-category clustering -> tf*idf keywords -> argmax
def cluster_labels(X, algorithm, eps, min_samples):
    """[V3-8] One place where the clustering algorithm is chosen."""
    if algorithm == 'hdbscan':
        if len(X) < max(2, min_samples) + 1:
            return np.full(len(X), -1)
        return HDBSCAN(min_cluster_size=max(2, min_samples),
                       metric='euclidean', copy=True).fit_predict(X)
    return DBSCAN(eps=eps, min_samples=min_samples,
                  metric='cosine').fit_predict(X)

def build_clusters(train, embedder, algorithm, eps, min_samples, top_k):
    by_cat = defaultdict(list)
    for v in train:
        by_cat[v['category']].append(v)

    # category-level document frequency, computed on TRAIN ONLY
    cat_df = category_document_frequency(by_cat)
    n_cat = len(by_cat)

    keywords = {}
    for cat, items in by_cat.items():
        if len(items) < 2:
            continue
        X = embedder.transform([v['clean'] for v in items])
        labels = cluster_labels(X, algorithm, eps, min_samples)
        by_lab = defaultdict(list)
        for v, lab in zip(items, labels):
            if lab != -1:
                by_lab[lab].append(v)
        cat_kw = {}
        for lab, group in by_lab.items():
            tf = Counter()
            for v in group:
                for w in v['clean'].split():
                    if w not in STOP_WORDS and len(w) > 2:
                        tf[w] += 1
            top = keyword_score(tf, cat_df, n_cat, top_k)
            if top:
                cat_kw[lab] = top
        if cat_kw:
            keywords[cat] = cat_kw
    return keywords


def predict_keyword(test, keywords, majority):
    preds, fallbacks, fired_idx = [], 0, []
    for v in test:
        toks = set(v['clean'].split())
        best, best_score = None, 0
        for cat, clusters in sorted(keywords.items()):
            score = max((len(toks & set(kw)) for kw in clusters.values()),
                        default=0)
            if score > best_score:
                best, best_score = cat, score
        if best_score > 0:
            preds.append(best)
            fired_idx.append(len(preds) - 1)
        else:
            preds.append(majority)
            fallbacks += 1
    return preds, (fallbacks / len(test) if test else 0.0), fired_idx

# one account, one seed
def run_split(videos, seed, embedder_kind, algorithm, eps, min_samples, top_k):
    train, test = stratified_split(videos, 0.2, seed)
    if not test:
        return None
    emb = Embedder(embedder_kind).fit([v['clean'] for v in train])

    y_tr = [v['category'] for v in train]
    y_te = [v['category'] for v in test]
    majority = Counter(y_tr).most_common(1)[0][0]

    Xtr = emb.transform([v['clean'] for v in train])
    Xte = emb.transform([v['clean'] for v in test])

    rng = np.random.default_rng(seed)
    cats, cnts = zip(*Counter(y_tr).items())
    p = np.array(cnts, dtype=float) / sum(cnts)

    preds = {
        'majority': [majority] * len(test),
        'random': list(rng.choice(cats, size=len(test), p=p)),
    }

    kw = build_clusters(train, emb, algorithm, eps, min_samples, top_k)
    preds['keyword (proposed)'], fallback_rate, fired_idx = predict_keyword(
        test, kw, majority)

    cent = {c: normalize(Xtr[[i for i, y in enumerate(y_tr) if y == c]]
                         .mean(axis=0).reshape(1, -1))[0]
            for c in set(y_tr)}
    cc = sorted(cent)
    M = np.vstack([cent[c] for c in cc])
    preds['centroid'] = [cc[i] for i in (Xte @ M.T).argmax(axis=1)]

    k = min(5, len(train))
    preds['kNN'] = list(KNeighborsClassifier(n_neighbors=k, metric='cosine')
                        .fit(Xtr, y_tr).predict(Xte))

    if len(set(y_tr)) > 1:
        preds['logreg'] = list(LogisticRegression(max_iter=2000, C=10.0,
                                                  class_weight='balanced')
                               .fit(Xtr, y_tr).predict(Xte))

    rows = {}
    for name, yp in preds.items():
        rows[name] = {
            'acc': accuracy_score(y_te, yp),
            'bacc': balanced_accuracy_score(y_te, yp),
            'f1': f1_score(y_te, yp, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_te, yp),
        }
    rows['keyword (proposed)']['fallback'] = fallback_rate

    kw_pred = preds['keyword (proposed)']
    if fired_idx:
        yt = [y_te[i] for i in fired_idx]
        rows['keyword (proposed)']['acc_fired'] = accuracy_score(
            yt, [kw_pred[i] for i in fired_idx])
        rows['keyword (proposed)']['majority_on_fired'] = accuracy_score(
            yt, [majority] * len(yt))
    else:
        rows['keyword (proposed)']['acc_fired'] = float('nan')
        rows['keyword (proposed)']['majority_on_fired'] = float('nan')
    rows['keyword (proposed)']['coverage'] = 1.0 - fallback_rate
    return rows, len(train), len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedder', default='labse',
                    choices=['labse', 'tfidf', 'char'])
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--algorithm', default='hdbscan',
                    choices=['hdbscan', 'dbscan'])
    ap.add_argument('--eps', type=float, default=0.51,
                    help='DBSCAN only; ignored under --algorithm hdbscan')
    ap.add_argument('--min-samples', type=int, default=2)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--dedup-key', default='video_id',
                    choices=['video_id', 'title'])
    ap.add_argument('--users', nargs='+', required=True)
    ap.add_argument('--min-videos', type=int, default=30,
                    help='skip accounts with fewer usable videos')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    agg = defaultdict(lambda: defaultdict(list))
    per_user = {}

    for path in args.users:
        videos = load_user(path, args.dedup_key)
        if len(videos) < args.min_videos:
            print(f'[skip] {path}: only {len(videos)} usable videos')
            continue
        acc_by_method = defaultdict(lambda: defaultdict(list))
        for seed in range(args.seeds):
            out = run_split(videos, seed, args.embedder, args.algorithm,
                            args.eps, args.min_samples, args.top_k)
            if out is None:
                continue
            rows, n_tr, n_te = out
            for m, met in rows.items():
                for k, v in met.items():
                    acc_by_method[m][k].append(v)
        per_user[path] = {m: {k: float(np.mean(v)) for k, v in met.items()}
                          for m, met in acc_by_method.items()}
        for m, met in acc_by_method.items():
            for k, v in met.items():
                agg[m][k].append(float(np.mean(v)))   # account-level mean
        print(f'[ok] {path}: n={len(videos)}')

    print(f'\n=== embedder={args.embedder}  algorithm={args.algorithm}  '
          f'dedup={args.dedup_key}  seeds={args.seeds}  '
          f'accounts={len(per_user)} ===')
    print(f'{"method":22s} {"acc":>14s} {"bal.acc":>14s} {"macroF1":>14s} '
          f'{"kappa":>14s}')
    order = ['majority', 'random', 'keyword (proposed)', 'centroid', 'kNN',
             'logreg']
    for m in order:
        if m not in agg:
            continue
        cells = []
        for k in ['acc', 'bacc', 'f1', 'kappa']:
            v = np.array(agg[m][k])
            cells.append(f'{v.mean():.3f}±{v.std(ddof=1):.3f}')
        print(f'{m:22s} ' + ' '.join(f'{c:>14s}' for c in cells))

    kwagg = agg.get('keyword (proposed)', {})
    if 'fallback' in kwagg:
        fb = np.array(kwagg['fallback'])
        cov = np.array(kwagg['coverage'])
        af = np.array([v for v in kwagg.get('acc_fired', []) if v == v])
        mf = np.array([v for v in kwagg.get('majority_on_fired', []) if v == v])
        print('\n--- keyword method: coverage vs. precision within coverage ---')
        print(f'  fallback-to-majority : {fb.mean():.1%} ± {fb.std(ddof=1):.1%} '
              f'of test items')
        print(f'  coverage (fired)     : {cov.mean():.1%} ± {cov.std(ddof=1):.1%}')
        if len(af):
            print(f'  accuracy when fired  : {af.mean():.3f} ± {af.std(ddof=1):.3f}')
            print(f'  majority on same set : {mf.mean():.3f} ± {mf.std(ddof=1):.3f}')
            print(f'  lift within coverage : {(af - mf).mean():+.3f}  '
                  f'wins={int((af - mf > 0).sum())}/{len(af)}')
        print(f'  overall accuracy      : {np.mean(kwagg["acc"]):.3f}  '
              '(mixes fired predictions with fallbacks; do not attribute to '
              'the method)')

    print('\n--- per-account accuracy ---')
    for path, rows in per_user.items():
        line = '  '.join(f'{m.split()[0]}={rows[m]["acc"]:.3f}'
                         for m in order if m in rows)
        fbv = rows.get('keyword (proposed)', {}).get('fallback')
        extra = f'  fallback={fbv:.1%}' if fbv is not None else ''
        print(f'{path.split("/")[-1]:20s} {line}{extra}')

    # Paired significance tests across accounts. With n accounts the smallest
    # attainable two-sided Wilcoxon p is 2^-(n-1), so n=5 cannot reach p<0.05
    # (floor 0.0625). The floor is printed so that a non-significant result is
    # not misread as a weak effect.
    try:
        from scipy.stats import wilcoxon
        n_users = len(per_user)
        floor = 2.0 ** -(n_users - 1) if n_users > 1 else 1.0
        print(f'\n--- paired Wilcoxon across {n_users} accounts '
              f'(smallest attainable p = {floor:.4f}) ---')
        pairs = [('keyword (proposed)', 'majority'),
                 ('kNN', 'keyword (proposed)'),
                 ('logreg', 'keyword (proposed)'),
                 ('centroid', 'keyword (proposed)')]
        for metric in ['acc', 'bacc']:
            print(f'  [{metric}]')
            for a, b in pairs:
                if a not in agg or b not in agg:
                    continue
                x, y = np.array(agg[a][metric]), np.array(agg[b][metric])
                wins = int((x - y > 0).sum())
                try:
                    _, p = wilcoxon(x, y)
                except ValueError:
                    p = float('nan')
                print(f'    {a:20s} vs {b:20s} '
                      f'diff={np.mean(x - y):+.3f}  wins={wins}/{n_users}  '
                      f'p={p:.4f}')
    except ImportError:
        print('\n(scipy not available; skipping significance tests)')

    out = args.out or (f'results_{args.embedder}_{args.algorithm}_'
                       f'{args.dedup_key}.json')
    json.dump({'config': vars(args),
               'aggregate': {m: {k: list(v) for k, v in met.items()}
                             for m, met in agg.items()},
               'per_user': per_user}, open(out, 'w'), indent=2)
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
