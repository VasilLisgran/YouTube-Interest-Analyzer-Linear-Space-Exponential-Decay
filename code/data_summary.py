"""data_summary.py -- per-account counts behind Table 1 (usable videos, distinct like-dates, span)."""
import sys
import pipeline_common as pc
print(f'{"account":8s} {"usable":>7s} {"distinct dates":>15s} {"span (days)":>12s}')
for p in sys.argv[1:]:
    ev = pc.load_events(p, dedup_key='video_id', require_date=True)
    d = [e['date'] for e in ev]
    acct = p.split('/')[-1].replace('user_videos_', '').replace('.json', '')
    print(f'{acct:8s} {len(ev):7d} {len(set(d)):15d} {(max(d) - min(d)).days:12d}')
