"""zero_score_share.py -- share of the 17 registered categories with zero score at each origin.

At lambda = 1.0 and W = infinity (the no-decay baseline of the main experiment), counts the
categories that have no event at all before the origin. Those categories tie and are ordered by the
hash tie-break. Origins are the ones of chrono_split.py (disjoint, 60-day horizon, >= 30 train,
>= 10 test events).
  python3 zero_score_share.py ../data/user_videos_u2.json ../data/user_videos_u3.json
"""
import sys
import chrono_split as cs

shares, n_cat = [], len(cs.REGISTERED_CATEGORIES)
for path in sys.argv[1:]:
    ev = cs.load(path)
    cuts = cs.place_origins(ev, 'disjoint', 5, 60, 30, 10)
    s = [sum(1 for c in cs.REGISTERED_CATEGORIES if cs.build_profile(tr, T, 1.0).get(c, 0) == 0) / n_cat
         for T, tr, te in cuts]
    shares += s
    print(f'{path}: {len(s)} origins, mean share of categories with zero score {sum(s)/len(s):.3f}')
print(f'pooled over {len(shares)} origins: {sum(shares)/len(shares):.3f} of {n_cat} categories')
