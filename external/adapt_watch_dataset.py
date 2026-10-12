"""adapt_watch_dataset.py -- convert a watch-history dataset into the per-user JSON
schema that chrono_split.py already reads (user_videos_<id>.json).

Written before the real files were seen: column names are therefore arguments, and the
script prints what it found. It makes no analysis choice beyond PROTOCOL_external_replication.md
(first watch per video, exact-name category mapping, drop unmapped, report the drops).

  python3 adapt_watch_dataset.py --events watch.csv --meta videos.csv --out-dir users/ \
      --user-col user_id --video-col video_id --date-col watch_date \
      --meta-video-col video_id --meta-cat-col category --meta-dur-col duration
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from datetime import datetime

REGISTERED = ('Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals', 'Sports',
              'Short Movies', 'Travel & Events', 'Gaming', 'Videoblogging', 'People & Blogs',
              'Comedy', 'Entertainment', 'News & Politics', 'Howto & Style', 'Education',
              'Science & Technology', 'Nonprofits & Activism')


def parse_date(s):
    s = str(s).strip()
    for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
                '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(s[:len(datetime.now().strftime(fmt))], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00')).date()
    except ValueError:
        return None


def parse_duration(s):
    s = str(s).strip()
    if s.isdigit():
        return int(s)
    try:
        return int(float(s))
    except ValueError:
        pass
    # ISO 8601 PT#H#M#S
    if s.startswith('PT'):
        n, t, tot = '', {'H': 3600, 'M': 60, 'S': 1}, 0
        for ch in s[2:]:
            if ch.isdigit():
                n += ch
            elif ch in t and n:
                tot += int(n) * t[ch]; n = ''
        return tot
    return None


def read_table(path):
    if path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', required=True)
    ap.add_argument('--meta', help='separate metadata table; omit if events already carry category/duration')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--user-col', default='user_id')
    ap.add_argument('--video-col', default='video_id')
    ap.add_argument('--date-col', default='watch_date')
    ap.add_argument('--meta-video-col', default='video_id')
    ap.add_argument('--meta-cat-col', default='category')
    ap.add_argument('--meta-dur-col', default='duration')
    ap.add_argument('--category-map', help='optional JSON {dataset_name: registered_name}; '
                    'default is exact-name match only')
    ap.add_argument('--keep-all-watches', action='store_true', help='sensitivity S1')
    a = ap.parse_args()

    ev = read_table(a.events)
    print(f'events table: {len(ev)} rows; columns: {list(ev[0].keys()) if ev else []}')
    meta = {}
    if a.meta:
        for r in read_table(a.meta):
            meta[str(r[a.meta_video_col])] = (r.get(a.meta_cat_col), r.get(a.meta_dur_col))
        print(f'meta table: {len(meta)} videos')
    cmap = {c: c for c in REGISTERED}
    if a.category_map:
        cmap.update(json.load(open(a.category_map, encoding='utf-8')))

    drop = Counter(); dropped_cats = Counter()
    per_user = defaultdict(list)
    for r in ev:
        vid = str(r[a.video_col]); d = parse_date(r[a.date_col])
        if d is None:
            drop['bad_date'] += 1; continue
        cat, dur = meta.get(vid, (r.get(a.meta_cat_col), r.get(a.meta_dur_col))) if meta else \
            (r.get(a.meta_cat_col), r.get(a.meta_dur_col))
        if cat in (None, '', 'nan'):
            drop['no_category'] += 1; continue
        if cat not in cmap:
            drop['unmapped_category'] += 1; dropped_cats[cat] += 1; continue
        sec = parse_duration(dur) if dur not in (None, '', 'nan') else None
        if sec is None or sec < 0:
            drop['no_duration'] += 1; continue
        per_user[str(r[a.user_col])].append(
            {'video_id': vid, 'title': vid, 'category_id': None, 'category': cmap[cat],
             'liked_at': d.isoformat(), 'duration_seconds': sec, 'weight': 0})

    os.makedirs(a.out_dir, exist_ok=True)
    n_kept = 0
    for u, rows in per_user.items():
        rows.sort(key=lambda x: x['liked_at'])
        if not a.keep_all_watches:
            seen, uniq = set(), []
            for x in rows:
                if x['video_id'] not in seen:
                    seen.add(x['video_id']); uniq.append(x)
            drop['repeat_watch'] += len(rows) - len(uniq)
            rows = uniq
        n_kept += len(rows)
        with open(os.path.join(a.out_dir, f'user_videos_{u}.json'), 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False)
    print(f'users written: {len(per_user)}; events kept: {n_kept}')
    print('dropped rows by reason:', dict(drop))
    if dropped_cats:
        print('unmapped categories (name: rows):', dict(dropped_cats.most_common()))


if __name__ == '__main__':
    sys.exit(main())
