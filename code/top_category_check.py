"""top_category_check.py -- which category is ranked first, with and without decay?

Descriptive support for the mechanism discussed in the paper (Section 5.1):
for each qualifying like account, at each of the disjoint origins of
chrono_split.py, it reports the category ranked first under the lambda = 1.0
control and under lambda = 0.95 (W = infinity), the most frequent category in
the following 60-day test window, and category counts over the whole history.
It is a description of the rankings, not a test of the mechanism.
"""
import sys
from collections import Counter
import chrono_split as cs

for path in sys.argv[1:]:
    ev = cs.load(path, 'continuous_log')
    if len({e['date'] for e in ev}) < 20:
        print(f'[skip] {path}: fewer than 20 distinct dates'); continue
    cuts = cs.place_origins(ev, 'disjoint', 5, 60, 30, 10)
    if len(cuts) < 3:
        print(f'[skip] {path}: {len(cuts)} origins'); continue
    cats = list(cs.REGISTERED_CATEGORIES)
    top = {1.0: Counter(), 0.95: Counter()}
    changed = []
    test_top = Counter()
    for T, tr, te in cuts:
        r = {lam: cs.rank_categories(cs.build_profile(tr, T, lam, None), cats, str(T))[0]
             for lam in (1.0, 0.95)}
        for lam in r: top[lam][r[lam]] += 1
        if r[1.0] != r[0.95]: changed.append((str(T), r[1.0], r[0.95]))
        test_top[Counter(e['category'] for e in te).most_common(1)[0][0]] += 1
    name = path.split('/')[-1]
    print(f'--- {name}: {len(ev)} events, {len(cuts)} origins ---')
    print('  most frequent categories, whole history (count):', Counter(e['category'] for e in ev).most_common(3))
    print('  ranked first under control lambda=1.0 (origins):  ', top[1.0].most_common())
    print('  ranked first under lambda=0.95 (origins):         ', top[0.95].most_common())
    print('  most frequent category in the test window (origins):', test_top.most_common())
    print(f'  first-ranked category differs at {len(changed)} of {len(cuts)} origins:')
    for c in changed: print('    ', c)
