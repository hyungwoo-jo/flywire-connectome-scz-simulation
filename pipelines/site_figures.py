#!/usr/bin/env python3
"""Summary figures for the research page (steps 68-72), drawn from saved csv outputs only. Writes qc_reports/site/."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'qc_reports/site'
NAMES = ['BANC_right', 'BANC_left', 'FlyWire_right', 'FlyWire_left', 'hemibrain_right']
LAB = {'BANC_right': 'BANC R', 'BANC_left': 'BANC L', 'FlyWire_right': 'FlyWire R', 'FlyWire_left': 'FlyWire L',
       'hemibrain_right': 'hemibrain R'}
COL = dict(zip(NAMES, ['#1f5f99', '#5b9bd5', '#b5452f', '#e08b6f', '#3d8a4f']))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def tradeoff(dial):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for n in NAMES:
        q = dial[dial.dataset == n].sort_values('alpha')
        ax[0].plot(q.specific * 100, q.ident * 100, '-o', ms=4, color=COL[n], label=LAB[n])
        r = q[q.alpha == 0].iloc[0]
        ax[0].plot(r.specific * 100, r.ident * 100, '*', ms=14, color=COL[n], mec='k', mew=.6)
        ax[1].plot(q.alpha, q.R_NMF, '-o', ms=4, color=COL[n])
        ax[2].plot(q.alpha, q.dprime26 / q[q.alpha == 0].dprime26.iloc[0], '-o', ms=4, color=COL[n])
        ax[2].plot(q.alpha, q.acc26 / q[q.alpha == 0].acc26.iloc[0], '--', color=COL[n], alpha=.6)
    ax[0].set_xlabel('interference (specific leak, %)')
    ax[0].set_ylabel('single-trial identification (%)')
    ax[0].set_title('A. identification vs interference\n(star = real wiring; line runs to flat)')
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_xlabel('alpha (0 = real, 1 = equal KC input)')
    ax[1].set_ylabel('NMF recovery of glomerular axes')
    ax[1].set_title('B. interpretability is U-shaped')
    ax[2].axhline(1, color='k', lw=.6, ls=':')
    ax[2].set_xlabel('alpha')
    ax[2].set_ylabel('relative to real wiring')
    ax[2].set_title("C. d' (solid) rises, accuracy (dashed) falls")
    for a in ax[1:]:
        a.axvline(0, color='k', lw=.6, ls=':')
    fig.tight_layout()
    fig.savefig(OUT / 'heterogeneity_dial.png', dpi=150)
    plt.close(fig)


def joint(f, dial):
    real = dial[dial.alpha == 0].set_index('dataset')
    main = f[~f.shuffle]
    lams = sorted(main.lam.unique())
    xpos = {l: i for i, l in enumerate(lams)}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for n in NAMES:
        for s, ls in ((0., '-'), (1., ':')):
            q = main[(main.dataset == n) & (main.start == s)].sort_values('lam')
            ax[0].plot([xpos[l] for l in q.lam], q.cv_final, ls, marker='o', ms=3.5, color=COL[n],
                       label=LAB[n] if s == 0 else None)
        ax[0].axhline(real.loc[n, 'cv'], color=COL[n], lw=.7, alpha=.5)
        q = main[(main.dataset == n) & (main.start == 0.)]
        ax[1].scatter(q.specific_final * 100, q.ident_final * 100, s=14, color=COL[n])
        ax[1].scatter(real.loc[n, 'specific'] * 100, real.loc[n, 'ident'] * 100, marker='*', s=160, color=COL[n],
                      edgecolor='k', linewidth=.6)
        t = f[(f.dataset == n) & ~f.shuffle & (f.start == 0.) & (f.lam == 0.)].iloc[0]
        s_ = f[(f.dataset == n) & f.shuffle].iloc[0]
        c0 = json.loads(t.curve)
        c1 = json.loads(s_.curve)
        ax[2].plot([r['step'] for r in c0], [r['cv'] for r in c0], '-', color=COL[n])
        ax[2].plot([r['step'] for r in c1], [r['cv'] for r in c1], ':', color=COL[n])
    ax[0].set_xticks(range(len(lams)), [f'{l:g}' for l in lams])
    ax[0].set_yscale('log')
    ax[0].set_xlabel('interference penalty lambda')
    ax[0].set_ylabel('KC input-strength CV (log)')
    ax[0].set_title('A. final heterogeneity\n(solid: real start, dotted: flat start; thin line = real CV)')
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_xlabel('interference (specific leak, %)')
    ax[1].set_ylabel('identification (%)')
    ax[1].set_title('B. learned wiring (dots) beats real (stars)')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('training step')
    ax[2].set_ylabel('CV (log)')
    ax[2].set_title('C. task (solid) vs shuffled labels (dotted)')
    fig.tight_layout()
    fig.savefig(OUT / 'joint_objective.png', dpi=150)
    plt.close(fig)


def schematic():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis('off')
    boxes = [(0.2, 'ORN / PN\n~50 glomeruli', '#d9e6f2'), (3.4, 'Kenyon cells (KC)\n~2,000, 5-10% active', '#f6e0c8'),
             (7.2, 'MBON output\n(learned value)', '#d8ecd9')]
    for x, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.2), 2.4, 1.2, boxstyle='round,pad=0.08', fc=c, ec='#555'))
        ax.text(x + 1.2, 1.8, t, ha='center', va='center', fontsize=10)
    ax.annotate('', (3.35, 1.8), (2.65, 1.8), arrowprops=dict(arrowstyle='->', lw=1.6))
    ax.annotate('', (7.15, 1.8), (5.85, 1.8), arrowprops=dict(arrowstyle='->', lw=1.6))
    ax.text(3.0, 2.15, 'connectome\n(fixed wiring)', ha='center', fontsize=8.5)
    ax.text(6.5, 2.15, 'plastic\n(learning)', ha='center', fontsize=8.5)
    ax.add_patch(FancyBboxPatch((3.9, 0.15), 1.4, 0.55, boxstyle='round,pad=0.05', fc='#eee', ec='#555'))
    ax.text(4.6, 0.42, 'APL inhibition', ha='center', va='center', fontsize=9)
    ax.annotate('', (4.6, 1.15), (4.6, 0.75), arrowprops=dict(arrowstyle='-[', lw=1.2))
    ax.text(4.6, 3.05, 'hub KC: many inputs -> active for many odors and for noise', ha='center', fontsize=9, color='#a0522d')
    fig.tight_layout()
    fig.savefig(OUT / 'circuit.png', dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dial = pd.read_csv(ROOT / 'qc_reports/heterogeneity_training/dial.csv')
    schematic()
    tradeoff(dial)
    joint(pd.read_csv(ROOT / 'qc_reports/joint_objective/training.csv'), dial)
    print(sorted(p.name for p in OUT.glob('*.png')))


if __name__ == '__main__':
    main()
