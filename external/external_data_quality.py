"""external_data_quality.py -- data-quality summary of the watch-history replication (log 16).

Line 1 needs the selected watch log and the metadata export (meta.csv) and reports, per user,
the share of distinct videos for which the YouTube Data API returned no category.
Lines 2-5 need only the chrono_split.py result of the replication and report the absolute
user-mean acc@1 / recall@3 under the no-decay control, under lambda = 0.95 and for the random
ranking floor, and the number of users and origins analysed.

  python3 external_data_quality.py --log log_selected.csv --meta meta.csv \
          --cs results/chrono_split_external.json            > ../logs/16_external_data_quality.txt
  python3 external_data_quality.py --cs results/chrono_split_external.json   # lines 2-5 only

--by-year additionally prints the share of watch events without metadata per calendar year
(the figure quoted in Section 3.2.3 of the paper); see logs/16b_external_missing_by_year.txt.

The script that wrote the committed log 16 was not kept; this one was written afterwards to the
same definitions. Run on 2026-10-12 against the original log_selected.csv and meta.csv, it
reproduces all five lines of log 16; the --by-year output is logs/16b_external_missing_by_year.txt
(its counts add up to the 104,534 rows without metadata of 672,366 reported in log 12).
"""
import argparse, csv, json, sys
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--log', help='selected watch log (log_selected.csv)')
ap.add_argument('--meta', help='metadata export of fetch_video_metadata.py (meta.csv)')
ap.add_argument('--cs', required=True, help='chrono_split.py JSON of the replication')
ap.add_argument('--user-col', default='user_id')
ap.add_argument('--video-col', default='video_id')
ap.add_argument('--date-col', default='watch_date')
ap.add_argument('--by-year', action='store_true')
a = ap.parse_args()

if a.log and a.meta:
    has_meta = set()
    with open(a.meta, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if (r.get('category') or '').strip():
                has_meta.add(r['video_id'])
    vids = defaultdict(set)
    year_tot, year_miss = defaultdict(int), defaultdict(int)
    with open(a.log, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            v = r[a.video_col]
            vids[r[a.user_col]].add(v)
            y = str(r[a.date_col])[:4]
            year_tot[y] += 1
            year_miss[y] += v not in has_meta
    share = np.array([sum(v not in has_meta for v in s) / len(s) for s in vids.values()])
    q1, med, q3 = np.percentile(share, [25, 50, 75])
    print(f"share of each user's distinct videos without metadata: median {med:.3f}, "
          f"IQR {q1:.3f}-{q3:.3f}, min {share.min():.3f}, max {share.max():.3f}")
elif a.log or a.meta:
    sys.exit('--log and --meta must be given together')

res = json.load(open(a.cs))['per_account']
users = sorted(res)
for key, lab in (('1.0', 'lambda=1.0   '), ('0.95', 'lambda=0.95  '), ('random', 'lambda=random')):
    acc = np.mean([res[u]['means'][key]['acc@1'] for u in users])
    rec = np.mean([res[u]['means'][key]['recall@3'] for u in users])
    print(f'{lab} user-mean acc@1 {acc:.3f}  recall@3 {rec:.3f}')
print(f'users analysed {len(users)} origins {sum(res[u]["n_origins"] for u in users)}')

if a.by_year and a.log and a.meta:
    print('\nshare of watch events without metadata, by year:')
    for y in sorted(year_tot):
        print(f'  {y}: {year_miss[y] / year_tot[y]:.3f}  ({year_miss[y]} of {year_tot[y]})')
