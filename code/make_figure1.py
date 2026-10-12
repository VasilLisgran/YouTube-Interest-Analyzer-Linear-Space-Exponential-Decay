import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAMS = ['0.8', '0.9', '0.95', '0.98', '0.99', '0.995', '0.999', '1.0']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='chrono_split.json')
    ap.add_argument('--output', default='figure1.png')
    args = ap.parse_args()

    d = json.load(open(args.input))
    accts = list(d['per_account'])
    x = list(range(len(LAMS)))
    labels = ['0.80', '0.90', '0.95', '0.98', '0.99', '0.995', '0.999',
              '1.00\n(control)']

    fig, axes = plt.subplots(len(accts), 1, figsize=(5.0, 5.6), sharex=True)
    if len(accts) == 1:
        axes = [axes]

    for ax, a in zip(axes, accts):
        means = d['per_account'][a]['means']
        n_or = d['per_account'][a]['n_origins']
        span = d['per_account'][a]['span']
        for metric, marker, style in (('acc@1', 'o', '-'),
                                      ('recall@3', 's', '-')):
            y = [means[l][metric] for l in LAMS]
            ax.plot(x, y, style, marker=marker, ms=4.5, lw=1.4, label=metric,
                    color='#222222' if metric == 'acc@1' else '#777777')
            ax.axhline(means['1.0'][metric], ls=':', lw=0.9,
                       color='#222222' if metric == 'acc@1' else '#777777')
        ax.axvline(len(LAMS) - 1, ls='--', lw=0.8, color='#bbbbbb')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f'{a}  ({span} d span, {n_or} origins)', fontsize=11)
        ax.set_ylabel('score', fontsize=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    axes[-1].set_xlabel(r'retention coefficient $\lambda$', fontsize=10)
    axes[0].legend(fontsize=9, frameon=False, loc='center right')
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print('written:', args.output)


if __name__ == '__main__':
    main()
