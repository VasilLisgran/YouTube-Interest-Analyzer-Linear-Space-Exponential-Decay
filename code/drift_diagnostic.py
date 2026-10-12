"""drift_diagnostic.py -- exploratory, post hoc: is the next window closer to the recent past or
to the whole past?  At every origin of chrono_split.py, compare the category distribution of the
test window (T, T+60] with (a) the distribution of the whole training history and (b) that of the
last 60 days before T, by total-variation distance (0 = identical, 1 = disjoint). Unweighted counts.
A positive 'recent advantage' (TV_all - TV_recent) means the recent past predicts the next window
better, i.e. the setting in which forgetting should help.
  python3 drift_diagnostic.py --label likes ../data/user_videos_u2.json ../data/user_videos_u3.json
"""
import argparse, types
from collections import Counter
from datetime import timedelta
import numpy as np
import chrono_split as cs

ap = argparse.ArgumentParser(); ap.add_argument('--label', default=''); ap.add_argument('users', nargs='+')
a = ap.parse_args()
args = types.SimpleNamespace(origins='disjoint', splits=5, horizon_days=60, min_train=30, min_test=10)


def dist(evs):
    c = Counter(e['category'] for e in evs); n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


per_user = []
for path in a.users:
    ev = cs.load(path)
    if len({e['date'] for e in ev}) < 20:
        continue
    cuts = cs.place_origins(ev, args.origins, args.splits, args.horizon_days, args.min_train, args.min_test)
    if len(cuts) < 3:
        continue
    adv = []
    for T, train, test in cuts:
        recent = [e for e in train if e['date'] > T - timedelta(days=60)]
        if not recent:
            continue
        adv.append(tv(dist(test), dist(train)) - tv(dist(test), dist(recent)))
    if adv:
        per_user.append(np.mean(adv))
d = np.array(per_user)
b = d[np.random.default_rng(0).integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
lo, hi = np.percentile(b, [2.5, 97.5])
print(f'{a.label}: users {len(d)}; mean recent advantage (TV_all - TV_recent) {d.mean():+.3f} '
      f'[{lo:+.3f},{hi:+.3f}]; positive for {int((d > 0).sum())}/{len(d)}; per-user values: '
      + ('' if len(d) > 5 else ' '.join(f'{x:+.3f}' for x in d)))
