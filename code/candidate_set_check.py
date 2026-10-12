"""candidate_set_check.py -- does the choice of candidate set matter?

chrono_split.py ranks all 17 registered categories at every origin.  This script
re-runs the evaluation with the candidate set restricted to the categories that
occur anywhere in the account's history (a deliberately look-ahead choice, used
only as a sensitivity check) and reports the means next to the registry result.
"""
import sys, types
import numpy as np
import chrono_split as cs

users = sys.argv[1:] or ['../data/user_videos_u2.json', '../data/user_videos_u3.json']
args = types.SimpleNamespace(origins='disjoint', splits=5, horizon_days=60,
                             min_train=30, min_test=10, min_origins=3,
                             max_age_days=0, random_repeats=200)
GRID = [0.95, 1.0]
full = tuple(cs.REGISTERED_CATEGORIES)
print(f'{"account":8s} {"candidates":>10s} {"lambda":>7s} {"acc@1":>7s} {"recall@3":>9s} {"MRR":>7s} {"nDCG@5":>7s}')
for p in users:
    acct = p.split('/')[-1].replace('user_videos_', '').replace('.json', '')
    ev = cs.load(p, 'continuous_log')
    occ = tuple(c for c in full if c in {e['category'] for e in ev})
    for label, cands in (('registry', full), ('occurring', occ)):
        cs.REGISTERED_CATEGORIES = cands
        r = cs.evaluate_account(ev, GRID, args)
        for lam in ('0.95', '1.0', 'random'):
            m = r['means'][lam]
            print(f'{acct:8s} {label + "=" + str(len(cands)):>10s} {lam:>7s} '
                  f'{m["acc@1"]:7.3f} {m["recall@3"]:9.3f} {m["MRR"]:7.3f} {m["nDCG@5"]:7.3f}')
