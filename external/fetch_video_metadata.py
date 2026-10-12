"""fetch_video_metadata.py -- category and duration for every video in the watch log,
via the YouTube Data API v3 (videos.list, part=snippet,contentDetails).

Run on your own machine (needs internet access to googleapis.com and an API key):

    export YT_API_KEY=...            # an API key is enough; OAuth is not needed
    python3 fetch_video_metadata.py --log youtube_watch_log.csv --out meta.jsonl
    # ...repeat on the following days until it reports "all done"...
    python3 fetch_video_metadata.py --log youtube_watch_log.csv --out meta.jsonl --export meta.csv

Facts behind the design
- videos.list accepts up to 50 ids per call and costs 1 quota unit per call; the default project
  quota is 10,000 units/day, so 1,172,110 videos = 23,443 calls = about 3 days.
- The script is resumable: every answered id (found OR unavailable) is appended to --out at once,
  and on restart those ids are skipped. Stopping it (Ctrl+C, quota, power) loses at most one batch.
- Ids the API does not return (deleted, private, blocked) are recorded with status "unavailable",
  so their share can be reported, as the protocol requires.
- Only what the analysis needs is stored: id, categoryId, duration. No titles, no channel data.
  Check the YouTube API Services Developer Policies on storing API data before publishing it;
  the safe default is to publish video ids and results, not the fetched metadata.
- Standard library only (no pip installs needed).
"""
import argparse, csv, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request

API = 'https://www.googleapis.com/youtube/v3/videos'
# YouTube's standard category ids -> the 17 names registered in chrono_split.py.
# Ids 30-44 are deliberately absent, as in CategoryRegistry.java; such videos are kept in
# the export with their raw id so the adapter can count them as unmapped.
CATEGORY_NAMES = {
    '1': 'Film & Animation', '2': 'Autos & Vehicles', '10': 'Music', '15': 'Pets & Animals',
    '17': 'Sports', '18': 'Short Movies', '19': 'Travel & Events', '20': 'Gaming',
    '21': 'Videoblogging', '22': 'People & Blogs', '23': 'Comedy', '24': 'Entertainment',
    '25': 'News & Politics', '26': 'Howto & Style', '27': 'Education',
    '28': 'Science & Technology', '29': 'Nonprofits & Activism'}

_DUR = re.compile(r'^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$')


def iso_duration_seconds(s):
    """'PT1H2M3S' -> 3723; 'P1DT2H' -> 93600; None if unparsable (e.g. 'P0D' for live)."""
    m = _DUR.match(s or '')
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def read_ids(log_path, col):
    seen, ids = set(), []
    with open(log_path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            v = (r.get(col) or '').strip()
            if v and v not in seen:
                seen.add(v); ids.append(v)
    return ids


def read_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)['id'])
                    except (ValueError, KeyError):
                        pass   # a half-written last line after a crash; that id is refetched
    return done


class QuotaExceeded(Exception):
    pass


def call_api(ids, key, timeout=30, max_tries=6):
    q = urllib.parse.urlencode({
        'part': 'snippet,contentDetails', 'id': ','.join(ids), 'key': key, 'maxResults': 50,
        'fields': 'items(id,snippet/categoryId,contentDetails/duration)'})
    delay = 2.0
    for attempt in range(1, max_tries + 1):
        try:
            with urllib.request.urlopen(f'{API}?{q}', timeout=timeout) as r:
                return json.load(r).get('items', [])
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'replace')
            if e.code == 403 and ('quotaExceeded' in body or 'dailyLimitExceeded' in body):
                raise QuotaExceeded(body[:300])
            if e.code in (400, 401, 403):          # bad key / API not enabled: retrying will not help
                sys.exit(f'HTTP {e.code}: {body[:500]}')
            err = f'HTTP {e.code}'
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = repr(e)
        if attempt == max_tries:
            raise RuntimeError(f'giving up on a batch after {max_tries} tries: {err}')
        print(f'  transient error ({err}); retry {attempt}/{max_tries - 1} in {delay:.0f}s', flush=True)
        time.sleep(delay); delay = min(delay * 2, 120)


def export_csv(out_path, csv_path):
    n = {'ok': 0, 'unavailable': 0, 'no_duration': 0, 'unregistered_category': 0}
    with open(out_path, encoding='utf-8') as f, open(csv_path, 'w', encoding='utf-8', newline='') as g:
        w = csv.writer(g); w.writerow(['video_id', 'category', 'duration'])
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r['status'] != 'ok':
                n['unavailable'] += 1
                w.writerow([r['id'], '', '']); continue
            cat = CATEGORY_NAMES.get(r.get('categoryId') or '', f"id{r.get('categoryId')}")
            if cat.startswith('id'):
                n['unregistered_category'] += 1
            if r.get('duration_seconds') is None:
                n['no_duration'] += 1
            n['ok'] += 1
            w.writerow([r['id'], cat, '' if r.get('duration_seconds') is None else r['duration_seconds']])
    print(f'exported {csv_path}: {n}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='youtube_watch_log.csv')
    ap.add_argument('--out', default='meta.jsonl', help='append-only results, used for resuming')
    ap.add_argument('--video-col', default='video_id')
    ap.add_argument('--max-calls', type=int, default=9500,
                    help='stop after this many API calls in this run (default stays under 10,000/day)')
    ap.add_argument('--sleep', type=float, default=0.1, help='pause between calls, seconds')
    ap.add_argument('--export', help='also write video_id,category,duration CSV for the adapter')
    a = ap.parse_args()

    ids = read_ids(a.log, a.video_col)
    done = read_done(a.out)
    todo = [v for v in ids if v not in done]
    print(f'videos in log: {len(ids)}; already answered: {len(done)}; to fetch: {len(todo)} '
          f'(~{-(-len(todo) // 50)} calls)', flush=True)

    if todo:
        key = os.environ.get('YT_API_KEY')
        if not key:
            sys.exit('set the YT_API_KEY environment variable first')
        calls = 0
        with open(a.out, 'a', encoding='utf-8') as out:
            try:
                for i in range(0, len(todo), 50):
                    if calls >= a.max_calls:
                        print(f'reached --max-calls={a.max_calls}; run again tomorrow to continue')
                        break
                    batch = todo[i:i + 50]
                    items = call_api(batch, key)
                    calls += 1
                    got = {}
                    for it in items:
                        got[it['id']] = {
                            'id': it['id'], 'status': 'ok',
                            'categoryId': it.get('snippet', {}).get('categoryId'),
                            'duration_seconds': iso_duration_seconds(
                                it.get('contentDetails', {}).get('duration'))}
                    for v in batch:
                        out.write(json.dumps(got.get(v, {'id': v, 'status': 'unavailable'})) + '\n')
                    out.flush()
                    if calls % 100 == 0:
                        print(f'  {calls} calls, {min(i + 50, len(todo))}/{len(todo)} ids', flush=True)
                    time.sleep(a.sleep)
            except QuotaExceeded as e:
                print(f'daily quota exhausted after {calls} calls in this run; progress is saved. '
                      f'Run again after the quota resets (midnight Pacific time).\n{e}')
            except KeyboardInterrupt:
                print('\ninterrupted; progress is saved, run again to continue')
        remaining = len(ids) - len(read_done(a.out))
        print('all done' if remaining == 0 else f'remaining: {remaining} videos')

    if a.export:
        if len(ids) - len(read_done(a.out)) > 0:
            print('note: exporting a partial result; unanswered videos are simply absent')
        export_csv(a.out, a.export)


if __name__ == '__main__':
    main()
