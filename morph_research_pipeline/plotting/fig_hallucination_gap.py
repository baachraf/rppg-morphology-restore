import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
try:
    from .config import apply_rc
    from .utils import save_fig, save_raw_data
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc
    from morph_research_pipeline.plotting.utils import save_fig, save_raw_data


def plot_hallucination_gap(hall_df, out_dir, figsize=(14, 5)):
    apply_rc()
    hall_wide = hall_df.pivot_table(
        index=['sid', 'dataset'], columns='condition', values='r_gt_mean'
    ).reset_index()

    cond_real    = hall_wide['Real'].dropna()
    cond_noise   = hall_wide['Noise'].dropna()
    cond_shuffle = hall_wide['Shuffled'].dropna() if 'Shuffled' in hall_wide.columns else pd.Series(dtype=float)

    stat, p_mwu  = stats.mannwhitneyu(cond_real, cond_noise, alternative='greater')
    _, p_t       = stats.ttest_ind(cond_real, cond_noise)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: distributions
    bins = np.linspace(-0.2, 1.0, 30)
    axes[0].hist(cond_real,  bins=bins, alpha=0.75, color='#1f77b4',
                 label=f'Real  (mean={cond_real.mean():.3f})')
    axes[0].hist(cond_noise, bins=bins, alpha=0.75, color='#d62728',
                 label=f'Noise (mean={cond_noise.mean():.3f})')
    if len(cond_shuffle):
        axes[0].hist(cond_shuffle, bins=bins, alpha=0.60, color='#ff7f0e',
                     label=f'Shuffled (mean={cond_shuffle.mean():.3f})')
    axes[0].set_xlabel('Correlation with GT (r)')
    axes[0].set_ylabel('Count (subjects)')
    axes[0].set_title(f'(A) Hallucination gap distribution\nReal > Noise: p={p_mwu:.4f} (MWU)')
    axes[0].legend(fontsize=9)

    # Panel B: paired scatter
    shared = hall_wide.dropna(subset=['Real', 'Noise'])
    axes[1].scatter(shared['Noise'], shared['Real'], s=60, color='#1f77b4',
                    alpha=0.7, edgecolors='white', linewidths=0.5)
    lims = [shared[['Real', 'Noise']].min().min() - 0.05,
            shared[['Real', 'Noise']].max().max() + 0.05]
    axes[1].plot(lims, lims, 'k--', alpha=0.4, label='y = x (no gap)')
    above = (shared['Real'] > shared['Noise']).sum()
    axes[1].text(lims[0] + 0.05, lims[1] - 0.12,
                 f'{above}/{len(shared)} subjects\nabove diagonal',
                 fontsize=10, color='#1f77b4')
    axes[1].set_xlabel('r (Noise input)')
    axes[1].set_ylabel('r (Real rPPG input)')
    axes[1].set_title('(B) Paired: Real vs Noise\nAbove diagonal = signal is being read')
    axes[1].legend()

    # Panel C: mean ± SEM bar
    if len(cond_shuffle):
        conditions = ['Noise', 'Shuffled', 'Real']
        cond_data  = [cond_noise, cond_shuffle, cond_real]
        bar_cols   = ['#d62728', '#ff7f0e', '#1f77b4']
    else:
        conditions = ['Noise', 'Real']
        cond_data  = [cond_noise, cond_real]
        bar_cols   = ['#d62728', '#1f77b4']
    means = [d.mean() for d in cond_data]
    sems  = [d.sem()  for d in cond_data]
    axes[2].bar(conditions, means, yerr=sems, color=bar_cols, alpha=0.8,
                edgecolor='white', capsize=5)
    axes[2].set_ylabel('Mean correlation with GT (r)')
    axes[2].set_title('(C) Mean ± SEM by input condition\n(V6 architecture)')
    for i, (m_v, s_v) in enumerate(zip(means, sems)):
        axes[2].text(i, m_v + s_v + 0.01, f'{m_v:.3f}', ha='center', fontsize=11)
    axes[2].set_ylim(0, max(means) + 0.25)

    fig.suptitle('Hallucination Gap: V6 Reads the Camera Signal (p=0.031)\n'
                 'Necessary but not sufficient condition for subject-specific reconstruction',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = save_fig(fig, out_dir, 'hallucination_gap.png')

    # ── Individual panels (paper figures — no suptitle) ───────────────────────
    # Panel B: paired scatter — real rPPG vs noise per subject
    shared = hall_wide.dropna(subset=['Real', 'Noise'])
    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    ax_b.scatter(shared['Noise'], shared['Real'], s=60, color='#1f77b4',
                 alpha=0.7, edgecolors='white', linewidths=0.5)
    lims_b = [shared[['Real', 'Noise']].min().min() - 0.05,
              shared[['Real', 'Noise']].max().max() + 0.05]
    ax_b.plot(lims_b, lims_b, 'k--', alpha=0.4, label='y = x (no gap)')
    above_b = (shared['Real'] > shared['Noise']).sum()
    ax_b.text(lims_b[0] + 0.05, lims_b[1] - 0.12,
              f'{above_b}/{len(shared)} subjects\nabove diagonal',
              fontsize=10, color='#1f77b4')
    ax_b.set_xlabel('r (Noise input)')
    ax_b.set_ylabel('r (Real rPPG input)')
    ax_b.legend(fontsize=9)
    fig_b.tight_layout()
    save_fig(fig_b, out_dir, 'hallucination_gap_paired.png')
    plt.close(fig_b)

    # Panel C: mean ± SEM bar chart by condition (paper figure)
    if len(cond_shuffle):
        conditions_c = ['Noise', 'Shuffled', 'Real rPPG']
        cond_data_c  = [cond_noise, cond_shuffle, cond_real]
        bar_cols_c   = ['#d62728', '#ff7f0e', '#1f77b4']
    else:
        conditions_c = ['Noise', 'Real rPPG']
        cond_data_c  = [cond_noise, cond_real]
        bar_cols_c   = ['#d62728', '#1f77b4']
    means_c = [d.mean() for d in cond_data_c]
    sems_c  = [d.sem()  for d in cond_data_c]
    fig_c, ax_c = plt.subplots(figsize=(3.4, 1.7))
    ax_c.bar(conditions_c, means_c, yerr=sems_c, color=bar_cols_c, alpha=0.8,
             edgecolor='white', capsize=3)
    ax_c.set_ylabel('Mean $r$ with GT', fontsize=8)
    ax_c.tick_params(labelsize=7)
    ax_c.set_ylim(0, max(means_c) + max(sems_c) + 0.09)
    for i, (m_v, s_v) in enumerate(zip(means_c, sems_c)):
        ax_c.text(i, m_v + s_v + 0.01, f'{m_v:.3f}', ha='center', fontsize=6)
    fig_c.tight_layout(pad=0.4)
    save_fig(fig_c, out_dir, 'hallucination_gap_bar.png')
    plt.close(fig_c)

    paired = hall_wide.dropna(subset=['Real', 'Noise'])
    save_raw_data({
        'figure': 'fig3_hallucination_gap',
        'description': 'VAE-Orth per-subject r under real rPPG vs white noise vs shuffled',
        'n_subjects': int(len(cond_real)),
        'real_rppg': {
            'mean':   float(cond_real.mean()),
            'std':    float(cond_real.std()),
            'sem':    float(cond_real.sem()),
            'values': cond_real.tolist(),
        },
        'white_noise': {
            'mean':   float(cond_noise.mean()),
            'std':    float(cond_noise.std()),
            'sem':    float(cond_noise.sem()),
            'values': cond_noise.tolist(),
        },
        'shuffled': {
            'mean':   float(cond_shuffle.mean()) if len(cond_shuffle) else None,
            'std':    float(cond_shuffle.std())  if len(cond_shuffle) else None,
            'sem':    float(cond_shuffle.sem())  if len(cond_shuffle) else None,
            'values': cond_shuffle.tolist()       if len(cond_shuffle) else [],
        },
        'mannwhitneyu_real_greater_noise': {
            'statistic': float(stat), 'p_val': float(p_mwu),
        },
        'ttest_real_vs_noise': {'p_val': float(p_t)},
        'n_subjects_real_above_noise': int((paired['Real'] > paired['Noise']).sum()),
        'paper_values': {
            'reported_p_mwu':        0.031,
            'reported_real_mean_r':  0.613,
            'reported_noise_mean_r': 0.447,
        },
    }, out_dir, 'fig3_hallucination_gap.json')

    plt.show()
    print(f'Mann-Whitney U: Real > Noise  p={p_mwu:.4f}  |  Welch t: p={p_t:.4f}')
    return path


if __name__ == '__main__':
    import shutil
    from morph_research_pipeline.plotting.config import RES, FIGS
    _JOURNAL = Path(r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\Projects_Papers\Journals\TobeSubmitted\JBHI_Jrnl_0526\figures')
    hall = pd.read_csv(RES / 'v6' / 'hallucination_audit.csv')
    plot_hallucination_gap(hall, FIGS)
    shutil.copy2(FIGS / 'hallucination_gap_bar.png', _JOURNAL / 'fig5_hallucination_gap.png')
    print('fig5_hallucination_gap.png -> journal figures')
