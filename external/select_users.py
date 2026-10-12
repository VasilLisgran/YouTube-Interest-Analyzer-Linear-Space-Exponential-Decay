"""select_users.py -- pre-specified random subset of users that fits one day of YouTube API quota.

Rule (fixed in PROTOCOL_external_replication.md before any metadata is fetched):
  shuffle the user ids with numpy default_rng(SEED); add users in that order while the number of
  UNIQUE video ids across the selected users stays <= BUDGET (default 9500 calls x 50 = 475,000);
  stop at the first user that would exceed the budget. Outcomes (categories, results) play no role.

  python3 select_users.py --log youtube_watch_log.csv --out-log log_selected.csv --out-ids selected_users.txt
"""
import argparse, csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--log', required=True)
ap.add_argument('--out-log', default='log_selected.csv')
ap.add_argument('--out-ids', default='selected_users.txt')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=int, default=9500 * 50, help='max unique videos (API calls x 50)')
ap.add_argument('--user-col', default='user_id')
ap.add_argument('--video-col', default='video_id')
a = ap.parse_args()

vids = {}
with open(a.log, encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        vids.setdefault(r[a.user_col], set()).add(r[a.video_col])
users = sorted(vids, key=lambda u: (len(u), u))          # deterministic base order
order = [users[i] for i in np.random.default_rng(a.seed).permutation(len(users))]

chosen, seen = [], set()
for u in order:
    new = vids[u] - seen
    if len(seen) + len(new) > a.budget:
        break
    chosen.append(u); seen |= vids[u]
print(f'users in log: {len(users)}; selected: {len(chosen)}; unique videos: {len(seen)} '
      f'(~{-(-len(seen) // 50)} API calls); seed={a.seed}; budget={a.budget}')
print(f'not used (quota): {len(users) - len(chosen)} users')
keep = set(chosen)
with open(a.log, encoding='utf-8', newline='') as f, open(a.out_log, 'w', encoding='utf-8', newline='') as g:
    rd = csv.DictReader(f); w = csv.DictWriter(g, fieldnames=rd.fieldnames); w.writeheader()
    for r in rd:
        if r[a.user_col] in keep:
            w.writerow(r)
open(a.out_ids, 'w').write('\n'.join(chosen) + '\n')
