"""tie_sensitivity.py -- how much of the reported difference is tie-break noise?

Categories with no events share a score of zero and must be ordered somehow.
chrono_split.py breaks those ties with a hash, which is deterministic but
arbitrary: a different hash salt gives a different, equally valid ordering.
This script re-runs the confirmatory comparison under many salts and reports
the spread, so that the size of the effect can be compared against the size of
the arbitrariness.
"""
import math, sys
from collections import Counter
import numpy as np
import chrono_split as cs

users = sys.argv[1:] or ['data_real/user_videos_u2.json', 'data_real/user_videos_u3.json']
N_SALT = 40
W_LIST = [None, 360]
for W in W_LIST:
  out = {m: [] for m in cs.METRICS}
  for salt in range(N_SALT):
      cs.TIE_SALT = f's{salt}'
      diffs = {m: [] for m in cs.METRICS}
      for path in users:
          ev = cs.load(path, 'continuous_log')
          if len({e['date'] for e in ev}) < 20:
              continue
          cuts = cs.place_origins(ev, 'disjoint', 5, 60, 30, 10)
          if len(cuts) < 3:
              continue
          for T, tr, te in cuts:
              tg = [e['category'] for e in te]; cnt = Counter(tg)
              ideal = sorted(cnt.values(), reverse=True)[:5]
              idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
              sc = {}
              for lam in (0.95, 1.0):
                  r = cs.rank_categories(cs.build_profile(tr, T, lam, W),
                                         list(cs.REGISTERED_CATEGORIES), str(T))
                  sc[lam] = cs.score_ranking(r, tg, cnt, idcg)
              for m in cs.METRICS:
                  diffs[m].append(sc[0.95][m] - sc[1.0][m])
      for m in cs.METRICS:
          out[m].append(float(np.mean(diffs[m])))

  lab = 'W = infinity' if W is None else f'W = {W}'
  print(f'\nconfirmatory comparison (lambda = 0.95 vs control), {lab}, under '
        f'{N_SALT} independent tie-break salts, 28 origins each')
  print(f"{'metric':10s} {'mean':>9s} {'sd':>9s} {'min':>9s} {'max':>9s}  "
        f"{'salts>0':>9s}")
  for m in cs.METRICS:
      v = np.array(out[m])
      print(f'{m:10s} {v.mean():+9.4f} {v.std(ddof=1):9.4f} {v.min():+9.4f} '
            f'{v.max():+9.4f}  {int((v > 0).sum()):>6d}/{N_SALT}')
