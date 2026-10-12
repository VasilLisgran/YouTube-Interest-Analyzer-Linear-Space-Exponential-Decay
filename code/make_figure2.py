"""make_figure2.py -- difference from the lambda = 1.0 control across the lambda grid,
for the like histories (pooled origins, moving-block bootstrap, block 2) and the
watch histories (user-level means, bootstrap over users).

  python3 make_figure2.py --likes chrono_split.json --watches chrono_split_external.json --output figure2.png
"""
import argparse, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chrono_split as cs

ap = argparse.ArgumentParser()
ap.add_argument('--likes', required=True)
ap.add_argument('--watches', required=True)
ap.add_argument('--output', default='figure2.png')
a = ap.parse_args()
likes = json.load(open(a.likes))['per_account']
watch = json.load(open(a.watches))['per_account']
grid = sorted(float(k) for k in next(iter(likes.values()))['per_origin'] if k not in ('random', '1.0'))
hl = [math.log(2) / -math.log(l) for l in grid]


def likes_stats(lam, m):
    d = []
    for r in likes.values():
        d += list(np.array(r['per_origin'][str(lam)][m]) - np.array(r['per_origin']['1.0'][m]))
    lo, hi = cs.block_bootstrap_ci(d, 2)
    return np.mean(d), lo, hi


def watch_stats(lam, m):
    d = np.array([np.mean(np.array(r['per_origin'][str(lam)][m]) - np.array(r['per_origin']['1.0'][m]))
                  for r in watch.values()])
    b = d[np.random.default_rng(0).integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return d.mean(), lo, hi


BLUE, ORANGE = '#2a78d6', '#eb6834'
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
for ax, m, title in zip(axes, ('recall@3', 'acc@1'), ('recall@3', 'acc@1')):
    for fn, col, mk, lab, off in ((likes_stats, BLUE, 'o', f'likes: 2 accounts, 28 origins', 0.97),
                                  (watch_stats, ORANGE, 's', f'watches: {len(watch)} users', 1.03)):
        s = np.array([fn(l, m) for l in grid])
        x = np.array(hl) * off
        ax.errorbar(x, s[:, 0], yerr=[s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]], color=col, marker=mk,
                    ms=5, lw=1.6, capsize=2, elinewidth=1, label=lab)
    ax.axhline(0, color='#666666', lw=0.9, ls='--')
    ax.axvline(13.5, color='#999999', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xticks([3, 7, 14, 34, 69, 138, 693]); ax.set_xticklabels(['3', '7', '14', '34', '69', '138', '693'])
    ax.set_xlabel('half-life of the decay (days, log scale)')
    ax.set_title(f'{title}: difference from the no-decay control', fontsize=10)
    ax.grid(axis='y', color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel('mean difference (95% CI)')
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(a.output, dpi=200)
print('written', a.output)
