"""
decay_diagnostics.py — how many events actually drive each category's weight?

Motivation: a category's normalized coordinate can be high either because many
events sustain it, or because one recent/long event dominates it. Both produce
the same number in `printUserVector()`; only this diagnostic tells them apart.

Uses Kish's effective sample size: N_eff = (sum w_i)^2 / sum(w_i^2).
  - N_eff == n  when all weights are equal (every event contributes equally)
  - N_eff -> 1  when one weight dominates all others (profile = one video)

Reads the JSON schema JSON_Reader now emits: video_id, title, category_id,
category, liked_at, duration_seconds, weight. Recomputes the decayed
contribution weight * lambda**days_ago independently of the Java code, using
the same reference date and lambda used for that run (read from the matching
run_manifest_<user>.json if available, or passed explicitly).

"""
import argparse
from collections import defaultdict
from datetime import date

from pipeline_common import WEIGHT_SCHEMES, load_events

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--reference-date', required=True)
    ap.add_argument('--lambda', dest='lam', type=float, default=0.95)
    ap.add_argument('--max-days', type=int, default=360)
    ap.add_argument('--weight-scheme', default='continuous_log',
                    choices=sorted(WEIGHT_SCHEMES))
    ap.add_argument('--top-n', type=int, default=3,
                    help='show this many dominant events per category')
    args = ap.parse_args()

    ref = date.fromisoformat(args.reference_date)
    videos = load_events(args.input, weight_scheme=args.weight_scheme,
                         dedup_key='video_id')

    by_cat = defaultdict(list)
    skipped_no_date, skipped_out_of_window = 0, 0

    for v in videos:
        if v.get('date') is None or v.get('weight') is None:
            skipped_no_date += 1
            continue
        days_ago = (ref - v['date']).days
        if days_ago < 0 or days_ago > args.max_days:
            skipped_out_of_window += 1
            continue
        decayed = v['weight'] * (args.lam ** days_ago)
        by_cat[v['category']].append({
            'title': v['title'], 'days_ago': days_ago,
            'raw_weight': round(v['weight'], 2), 'decayed': decayed,
        })

    print(f'reference={ref}  lambda={args.lam}  max_days={args.max_days}  '
          f'weight_scheme={args.weight_scheme}')
    print(f'events skipped (no liked_at / weight): {skipped_no_date}')
    print(f'events skipped (out of window): {skipped_out_of_window}')
    print()
    header = f'{"category":22s} {"n":>4s} {"N_eff":>7s} {"top1_share":>11s} {"top3_share":>11s}'
    print(header)
    print('-' * len(header))

    rows = []
    for cat, events in sorted(by_cat.items(), key=lambda kv: -sum(e['decayed'] for e in kv[1])):
        weights = sorted((e['decayed'] for e in events), reverse=True)
        total = sum(weights)
        sq = sum(w * w for w in weights)
        n_eff = (total * total / sq) if sq > 0 else 0.0
        top1 = weights[0] / total if total > 0 else 0.0
        top3 = sum(weights[:3]) / total if total > 0 else 0.0
        rows.append((cat, len(events), n_eff, top1, top3, events))
        print(f'{cat:22s} {len(events):4d} {n_eff:7.2f} {top1:10.1%} {top3:10.1%}')

    print()
    print('Categories where N_eff < 1.5 are effectively single-event profiles.')
    print('Dominant events per category:')
    for cat, n, n_eff, top1, top3, events in rows:
        if n_eff >= 1.5:
            continue
        events_sorted = sorted(events, key=lambda e: -e['decayed'])
        print(f'\n  [{cat}]  N_eff={n_eff:.2f}  n={n}')
        for e in events_sorted[:args.top_n]:
            print(f'    {e["decayed"]:8.2f}  ({e["days_ago"]:3d}d ago, '
                  f'raw_weight={e["raw_weight"]})  {e["title"][:70]}')


if __name__ == '__main__':
    main()