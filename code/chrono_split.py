"""
chrono_split.py -- does exponential temporal decay actually help predict
future likes?

This is the paper's decisive experiment. N_eff measures how concentrated a
profile is and increases monotonically as lambda -> 1, so it is maximised by
having NO decay at all; it reports the cost of decay and says nothing about its
benefit. Only a predictive task can weigh the two.

Protocol (rolling origin / forward chaining):

  for each origin T:
      profile  = sum over likes with date <= T of  w(duration) * lambda^(T-date)
                 restricted to ages <= W
      ranking  = categories sorted by profile score
      targets  = categories of the likes in (T, T + horizon]
      score the ranking against those targets

lambda = 1.0 is in the grid on purpose: it is the no-decay control. If no
lambda < 1.0 beats lambda = 1.0, the decay component is not earning its place.

--------------------------------------------------------------------------
Changes in this version (v3), all of them responses to referee-style
objections raised against v2:

  - ONE deduplication key across the whole pipeline: video_id, falling
         back to the title only when no id is present.  evaluate_v2.py and
         clusters.py now use the identical loader semantics, so the account
         sizes quoted anywhere in the paper come from a single convention.

  - Weights are RECOMPUTED from duration_seconds through an explicit
         weight scheme rather than read from the 'weight' field the Java
         client serialised.  The serialised field is the ORIGINAL scheme
         rounded to an integer; recomputing makes the scheme an experimental
         factor (--weight-scheme) instead of a property of the data file, so
         the whole experiment can be repeated under continuous_log and
         uniform.  ORIGINAL is retained only as a historical baseline.

  - The truncation age W is an explicit factor (--max-age-days).  The
         deployed Java client uses 360; the analysis defaults to no
         truncation so that lambda = 1.0 is a genuine no-forgetting control
         and not a sliding window.  Both settings are runnable, so the
         difference can be measured rather than argued about.

  - Origin placement mode.  'disjoint' (default) places origins one
         horizon apart, so that no two evaluation windows share a single
         like; this makes the origin, not the account, a usable unit of
         analysis.  'fraction' reproduces the earlier protocol (n_splits
         origins at evenly spaced fractions of the calendar span) and is
         kept so that the previously published numbers remain reproducible.

  - Accounts must supply at least --min-origins usable origins.  Under
         'disjoint' this is a mechanical consequence of the horizon and is
         fixed before any result is seen; it replaces the earlier post-hoc
         'decisive subset' argument.

  - Per-origin differences against the control are retained and tested:
         exact sign test and Wilcoxon signed-rank over origins, plus a
         moving-block bootstrap CI (block length --block) because
         consecutive origins share training data even when their evaluation
         windows are disjoint.

  - The random-ranking floor is averaged over --random-repeats shuffles
         per origin instead of a single draw, so the floor is not itself a
         high-variance quantity.

Usage:
    python3 chrono_split.py --users data/user_videos_u*.json
    python3 chrono_split.py --users ... --origins fraction --splits 5   # legacy
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np

# --------------------------------------------------------------------------
# duration -> weight schemes, mirroring
# YouTubeDataLoader.WeightScheme one-for-one.
# --------------------------------------------------------------------------
WEIGHT_SCHEMES = {
    # historical pilot scheme; discontinuous at 60 s (10 -> 101.6)
    'original': lambda s: 10.0 if s <= 60 else 40.0 + 15.0 * math.log(s),
    # continuous replacement
    'continuous_log': lambda s: 15.0 * math.log(1.0 + max(s, 1)),
    # duration ignored: the null hypothesis for duration weighting
    'uniform': lambda s: 1.0,
}

PLACEHOLDER_TITLES = ('private video', 'deleted video')

# The candidate set ranked at every origin. Fixing it to the registry
# rather than deriving it from the account's own history removes a subtle
# look-ahead: a set derived from all of an account's events encodes the fact
# that certain categories never occur, including in the not-yet-seen test
# windows. The leak does not bias the decay-vs-control comparison, since the
# same set is used under every lambda, but it inflates the random-ranking floor
# (1/15 instead of 1/17) and it is exactly the kind of thing this paper
# criticises elsewhere. Ranking all 17 registered categories at every origin
# leaves acc@1 and recall@3 unchanged to three decimals and lowers the floor to
# its correct level.
REGISTERED_CATEGORIES = (
    'Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals', 'Sports',
    'Short Movies', 'Travel & Events', 'Gaming', 'Videoblogging',
    'People & Blogs', 'Comedy', 'Entertainment', 'News & Politics',
    'Howto & Style', 'Education', 'Science & Technology',
    'Nonprofits & Activism')


def load(path, weight_scheme='continuous_log'):
    """Single deduplication convention, weights recomputed."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    wfun = WEIGHT_SCHEMES[weight_scheme]
    out, seen = [], set()
    for v in raw:
        title = (v.get('title') or '').strip()
        if not title or title.lower() in PLACEHOLDER_TITLES:
            continue
        if not v.get('liked_at'):
            continue
        key = v.get('video_id') or title
        if key in seen:
            continue
        seen.add(key)
        seconds = v.get('duration_seconds')
        if seconds is None or seconds < 0:
            continue
        out.append({'date': date.fromisoformat(v['liked_at']),
                    'weight': float(wfun(int(seconds))),
                    'category': v['category']})
    out.sort(key=lambda e: e['date'])
    return out


def build_profile(train, split_date, lam, max_age=None):
    """R_c = sum of w * lambda^age over events with 0 <= age <= W."""
    score = defaultdict(float)
    for e in train:
        age = (split_date - e['date']).days
        if age < 0:
            continue
        if max_age is not None and age > max_age:
            continue
        score[e['category']] += e['weight'] * (lam ** age)
    return score


TIE_SALT = ''   # varied by tie_sensitivity.py to probe tie-break noise


def _tie_key(origin_key, category):
    """A stable pseudo-random key for breaking ties between categories
    that hold the same score -- in practice the categories with no events at
    all, which all sit at 0.

    Deriving it from a hash of (origin, category) rather than from a shared
    RNG stream makes the ranking independent of how many times any other part
    of the program happened to draw from that stream. The earlier version drew
    the jitter from the caller's RNG, so two scripts that evaluated the same
    configuration but issued a different number of draws (chrono_split.py runs
    an eight-value lambda grid and 200 shuffles per origin; window_control.py
    runs six conditions and none) broke ties differently and reported MRR
    values that differed by 0.001. Nothing was wrong with either number, but a
    quantity that changes with call order is not reproducible, and the metrics
    most exposed to it -- MRR and nDCG@5, which read positions below the top of
    the ranking -- are exactly the ones this paper reports as null. Hashing
    removes the dependence: the same origin and category always produce the
    same key, in any script, in any order, in any process.
    """
    h = hashlib.blake2b(f'{TIE_SALT}|{origin_key}|{category}'.encode(),
                        digest_size=8).digest()
    return int.from_bytes(h, 'big')


def rank_categories(score, all_cats, origin_key):
    """Descending by score; ties broken by a stable hash so that neither
    alphabetical order nor RNG call history can leak an advantage."""
    return sorted(all_cats,
                  key=lambda c: (-score.get(c, 0.0), _tie_key(origin_key, c)))


def score_ranking(ranking, targets, counts, idcg, k_recall=3, k_ndcg=5):
    pos = {c: i for i, c in enumerate(ranking)}
    hits1 = hits_k = 0
    rr = []
    for c in targets:
        i = pos.get(c)
        if i is None:
            rr.append(0.0)
            continue
        if i == 0:
            hits1 += 1
        if i < k_recall:
            hits_k += 1
        rr.append(1.0 / (i + 1))
    gains = [counts.get(c, 0) for c in ranking[:k_ndcg]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n = len(targets)
    return {
        'acc@1': hits1 / n,
        f'recall@{k_recall}': hits_k / n,
        'MRR': float(np.mean(rr)),
        f'nDCG@{k_ndcg}': (dcg / idcg) if idcg > 0 else 0.0,
    }


METRICS = ['acc@1', 'recall@3', 'MRR', 'nDCG@5']


def place_origins(events, mode, n_splits, horizon_days, min_train, min_test):
    """Returns a list of (T, train, test)."""
    dates = [e['date'] for e in events]
    first, last = dates[0], dates[-1]
    hz = timedelta(days=horizon_days)

    if mode == 'fraction':
        cands = [first + (last - first) * float(f)
                 for f in np.linspace(0.5, 0.85, n_splits)]
    else:
        # one horizon apart: consecutive evaluation windows are
        # disjoint, so no like is scored at two origins.
        if len(events) < min_train:
            return []
        warm = dates[min_train - 1]   # earliest date with min_train events behind it
        cands, T = [], warm
        while T + hz <= last:
            cands.append(T)
            T = T + hz

    cuts = []
    for T in cands:
        train = [e for e in events if e['date'] <= T]
        test = [e for e in events if T < e['date'] <= T + hz]
        if len(train) >= min_train and len(test) >= min_test:
            cuts.append((T, train, test))
    return cuts


def evaluate_account(events, lam_grid, args, seed=0):
    rng = np.random.default_rng(seed)
    all_cats = list(REGISTERED_CATEGORIES)   # no look-ahead
    dates = [e['date'] for e in events]
    span = (dates[-1] - dates[0]).days
    if span <= 0:
        return None

    cuts = place_origins(events, args.origins, args.splits, args.horizon_days,
                         args.min_train, args.min_test)
    if len(cuts) < args.min_origins:
        return {'n_origins': len(cuts), 'span': span, 'ok': False}

    max_age = None if args.max_age_days <= 0 else args.max_age_days
    per_origin = {lam: {m: [] for m in METRICS} for lam in lam_grid}
    per_origin['random'] = {m: [] for m in METRICS}

    for T, train, test in cuts:
        # both the tie-break and the shuffles are keyed on the origin,
        # so neither depends on the order in which origins are processed.
        origin_key = str(T)
        rng = np.random.default_rng(
            int.from_bytes(hashlib.blake2b(origin_key.encode(),
                                           digest_size=8).digest(), 'big')
            % (2 ** 32))
        targets = [e['category'] for e in test]
        counts = Counter(targets)
        ideal = sorted(counts.values(), reverse=True)[:5]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        for lam in lam_grid:
            score = build_profile(train, T, lam, max_age)
            ranking = rank_categories(score, all_cats, origin_key)
            for m, v in score_ranking(ranking, targets, counts, idcg).items():
                per_origin[lam][m].append(v)
        # averaged random floor
        acc = {m: [] for m in METRICS}
        for _ in range(args.random_repeats):
            shuffled = list(all_cats)
            rng.shuffle(shuffled)
            for m, v in score_ranking(shuffled, targets, counts, idcg).items():
                acc[m].append(v)
        for m in METRICS:
            per_origin['random'][m].append(float(np.mean(acc[m])))

    means = {k: {m: float(np.mean(v)) for m, v in d.items()}
             for k, d in per_origin.items()}
    return {'ok': True, 'span': span, 'n_origins': len(cuts),
            'origins': [str(T) for T, _, _ in cuts],
            'n_test': [len(t) for _, _, t in cuts],
            'per_origin': {str(k): d for k, d in per_origin.items()},
            'means': {str(k): d for k, d in means.items()},
            '_means_raw': means}


# --------------------------------------------------------------------------
# statistics over origins
# --------------------------------------------------------------------------
def sign_test_p(diffs):
    """Two-sided exact binomial sign test, zeros discarded."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for d in nz if d > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail), k, n


def block_bootstrap_ci(diffs, block, reps=10000, seed=0):
    """Moving-block bootstrap CI for the mean of a dependent sequence."""
    x = np.asarray(diffs, dtype=float)
    n = len(x)
    if n == 0:
        return (float('nan'), float('nan'))
    b = max(1, min(block, n))
    n_blocks = int(math.ceil(n / b))
    starts_max = n - b + 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, starts_max, size=(reps, n_blocks))
    offs = np.arange(b)
    samples = (idx[:, :, None] + offs[None, None, :]).reshape(reps, -1)[:, :n]
    means = x[samples].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def origin_stats(res, lam, metric, block, seed=0):
    a = np.array(res['per_origin'][str(lam)][metric], dtype=float)
    b = np.array(res['per_origin']['1.0'][metric], dtype=float)
    d = a - b
    p_sign, k, n = sign_test_p(list(d))
    lo, hi = block_bootstrap_ci(d, block, seed=seed)
    try:
        from scipy.stats import wilcoxon
        p_w = float(wilcoxon(a, b).pvalue) if np.any(d) else 1.0
    except Exception:
        p_w = float('nan')
    return {'mean': float(d.mean()), 'better': k, 'nonzero': n,
            'n': len(d), 'p_sign': p_sign, 'p_wilcoxon': p_w,
            'ci_lo': lo, 'ci_hi': hi}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', nargs='+', required=True)
    ap.add_argument('--lambda-grid',
                    default='0.80,0.90,0.95,0.98,0.99,0.995,0.999,1.0')
    ap.add_argument('--weight-scheme', default='continuous_log',
                    choices=sorted(WEIGHT_SCHEMES))
    ap.add_argument('--max-age-days', type=int, default=0,
                    help='truncation age W in days; <=0 means no truncation')
    ap.add_argument('--origins', default='disjoint',
                    choices=['disjoint', 'fraction'])
    ap.add_argument('--splits', type=int, default=5,
                    help='number of origins in --origins fraction mode')
    ap.add_argument('--horizon-days', type=int, default=60)
    ap.add_argument('--min-train', type=int, default=30)
    ap.add_argument('--min-test', type=int, default=10)
    ap.add_argument('--min-origins', type=int, default=3)
    ap.add_argument('--min-distinct-dates', type=int, default=20)
    ap.add_argument('--random-repeats', type=int, default=200)
    ap.add_argument('--block', type=int, default=2,
                    help='moving-block bootstrap block length over origins')
    ap.add_argument('--focus-lambda', type=float, default=0.95)
    ap.add_argument('--out', default='chrono_split.json')
    args = ap.parse_args()

    lam_grid = [float(x) for x in args.lambda_grid.split(',')]
    per_account, skipped = {}, []

    print(f'weight_scheme={args.weight_scheme}  '
          f'W={"inf" if args.max_age_days <= 0 else args.max_age_days}  '
          f'origins={args.origins}  horizon={args.horizon_days}d  '
          f'min_origins={args.min_origins}')

    for path in args.users:
        acct = path.split('/')[-1].replace('user_videos_', '').replace('.json', '')
        events = load(path, args.weight_scheme)
        distinct = len({e['date'] for e in events})
        if distinct < args.min_distinct_dates:
            skipped.append((acct, len(events), distinct, 'distinct-dates'))
            print(f'[skip] {acct}: {len(events)} events on only {distinct} '
                  f'distinct dates -- no temporal signal')
            continue
        res = evaluate_account(events, lam_grid, args)
        if res is None or not res.get('ok'):
            n_o = 0 if res is None else res['n_origins']
            skipped.append((acct, len(events), distinct, f'{n_o} usable origins'))
            print(f'[skip] {acct}: {len(events)} events, span '
                  f'{"?" if res is None else res["span"]}d, only {n_o} usable '
                  f'origins (< {args.min_origins} required)')
            continue

        per_account[acct] = res
        print(f'\n--- {acct} ---  {len(events)} events, {distinct} distinct '
              f'dates, span {res["span"]}d, {res["n_origins"]} origins, '
              f'test sizes {res["n_test"]}')
        print(f'{"lambda":>9s} {"half-life":>10s} {"acc@1":>8s} {"recall@3":>9s} '
              f'{"MRR":>7s} {"nDCG@5":>8s}')
        for lam in lam_grid:
            r = res['_means_raw'][lam]
            hl = 'inf' if lam >= 1.0 else f'{math.log(2) / -math.log(lam):.1f}d'
            print(f'{lam:9.3f} {hl:>10s} {r["acc@1"]:8.3f} {r["recall@3"]:9.3f} '
                  f'{r["MRR"]:7.3f} {r["nDCG@5"]:8.3f}')
        r = res['_means_raw']['random']
        print(f'{"random":>9s} {"-":>10s} {r["acc@1"]:8.3f} {r["recall@3"]:9.3f} '
              f'{r["MRR"]:7.3f} {r["nDCG@5"]:8.3f}')

    if not per_account:
        print('\nNo account met the protocol requirements.')
        return

    # ---------------- aggregate across accounts ----------------
    print(f'\n=== aggregate across {len(per_account)} accounts ===')
    print(f'{"lambda":>9s} {"acc@1":>16s} {"recall@3":>16s} {"MRR":>16s} '
          f'{"nDCG@5":>16s}')
    agg = {}
    for lam in lam_grid + ['random']:
        cells, row = [], {}
        for metric in METRICS:
            vals = np.array([per_account[a]['_means_raw'][lam][metric]
                             for a in per_account])
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[metric] = (float(vals.mean()), sd)
            cells.append(f'{vals.mean():.3f}±{sd:.3f}')
        agg[str(lam)] = row
        label = f'{lam:9.3f}' if lam != 'random' else f'{"random":>9s}'
        print(label + ' ' + ' '.join(f'{c:>16s}' for c in cells))

    print('\n--- decay vs. no decay (lambda = 1.0 control), account-level ---')
    verdicts = {}
    for metric in METRICS:
        base = np.array([per_account[a]['_means_raw'][1.0][metric]
                         for a in per_account])
        best_lam, best_diff = None, -1e9
        for lam in lam_grid:
            if lam >= 1.0:
                continue
            v = np.array([per_account[a]['_means_raw'][lam][metric]
                          for a in per_account])
            d = float((v - base).mean())
            if d > best_diff:
                best_lam, best_diff = lam, d
        wins = int((np.array([per_account[a]['_means_raw'][best_lam][metric]
                              for a in per_account]) - base > 0).sum())
        verdicts[metric] = {'best_lambda': best_lam, 'diff': best_diff,
                            'wins': wins, 'n': len(per_account)}
        verdict = ('decay helps' if best_diff > 0
                   else 'NO decay setting beats lambda=1.0')
        print(f'  {metric:9s}: best lambda={best_lam}  diff={best_diff:+.4f}  '
              f'wins={wins}/{len(per_account)}  -> {verdict}')

    # ---------------- per-origin tests ----------------
    print(f'\n--- per-origin paired comparison, lambda={args.focus_lambda} '
          f'vs control (moving-block bootstrap, block={args.block}) ---')
    print(f'{"account":8s} {"metric":9s} {"n":>3s} {"mean d":>8s} '
          f'{"better":>9s} {"95% CI":>19s} {"p_sign":>8s} {"p_wilc":>8s}')
    stats = {}
    for acct, res in per_account.items():
        stats[acct] = {}
        for metric in METRICS:
            s = origin_stats(res, args.focus_lambda, metric, args.block)
            stats[acct][metric] = s
            ci = f'[{s["ci_lo"]:+.3f},{s["ci_hi"]:+.3f}]'
            print(f'{acct:8s} {metric:9s} {s["n"]:3d} {s["mean"]:+8.3f} '
                  f'{s["better"]:4d}/{s["nonzero"]:<4d} {ci:>19s} '
                  f'{s["p_sign"]:8.4f} {s["p_wilcoxon"]:8.4f}')

    def pooled_diffs(lam, metric):
        d = []
        for res in per_account.values():
            a = np.array(res['per_origin'][str(lam)][metric])
            b = np.array(res['per_origin']['1.0'][metric])
            d.extend(list(a - b))
        return d

    print('\n--- pooled over all origins of all qualifying accounts ---')
    pooled = {}
    for metric in METRICS:
        d = pooled_diffs(args.focus_lambda, metric)
        p, k, n = sign_test_p(d)
        lo, hi = block_bootstrap_ci(d, args.block)
        pooled[metric] = {'mean': float(np.mean(d)), 'better': k, 'nonzero': n,
                          'n': len(d), 'p_sign': p, 'ci_lo': lo, 'ci_hi': hi}
        print(f'  {metric:9s} n={len(d):3d}  mean={np.mean(d):+.3f}  '
              f'better={k}/{n}  95% CI [{lo:+.3f},{hi:+.3f}]  p_sign={p:.4f}')

    # full lambda grid, pooled per-origin mean difference and CI
    print('\n--- pooled per-origin mean difference vs control, whole grid ---')
    print(f'{"lambda":>8s} ' + ' '.join(f'{m:>24s}' for m in METRICS))
    grid_tab = {}
    for lam in lam_grid:
        if lam >= 1.0:
            continue
        cells, row = [], {}
        for metric in METRICS:
            d = pooled_diffs(lam, metric)
            lo, hi = block_bootstrap_ci(d, args.block)
            row[metric] = {'mean': float(np.mean(d)), 'ci_lo': lo, 'ci_hi': hi}
            cells.append(f'{np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]')
        grid_tab[str(lam)] = row
        print(f'{lam:8.3f} ' + ' '.join(f'{c:>24s}' for c in cells))

    # block-length sensitivity for the focus lambda
    print(f'\n--- block-length sensitivity, lambda={args.focus_lambda} ---')
    print(f'{"block":>6s} ' + ' '.join(f'{m:>24s}' for m in METRICS))
    block_tab = {}
    for b in (1, 2, 3, 4):
        cells, row = [], {}
        for metric in METRICS:
            d = pooled_diffs(args.focus_lambda, metric)
            lo, hi = block_bootstrap_ci(d, b)
            row[metric] = {'ci_lo': lo, 'ci_hi': hi}
            cells.append(f'{np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]')
        block_tab[str(b)] = row
        print(f'{b:6d} ' + ' '.join(f'{c:>24s}' for c in cells))

    json.dump({'config': vars(args),
               'per_account': {a: {k: v for k, v in d.items()
                                   if k != '_means_raw'}
                               for a, d in per_account.items()},
               'aggregate': agg, 'verdicts': verdicts,
               'origin_stats': stats, 'pooled_origin_stats': pooled,
               'lambda_grid_origin_stats': grid_tab,
               'block_sensitivity': block_tab,
               'skipped': skipped},
              open(args.out, 'w'), indent=2)
    print(f'\nwritten: {args.out}')


if __name__ == '__main__':
    main()
