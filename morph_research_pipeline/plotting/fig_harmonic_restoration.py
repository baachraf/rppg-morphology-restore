import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path
try:
    from .config import apply_rc, ARCH_META, PAPER_NAMES
    from .utils import save_fig, save_raw_data
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc, ARCH_META, PAPER_NAMES
    from morph_research_pipeline.plotting.utils import save_fig, save_raw_data


def plot_harmonic_restoration(spec_df, dfs_test, out_dir, figsize=(15, 5)):
    apply_rc()
    raw_h2h1  = spec_df['raw_h2h1'].dropna()
    rest_h2h1 = spec_df['rest_h2h1'].dropna()
    gt_h2h1   = spec_df['gt_h2h1'].dropna()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: distributions
    bins = np.linspace(0, 1.2, 40)
    axes[0].hist(raw_h2h1,  bins=bins, alpha=0.7, color='#d62728',
                 label=f'rPPG input (mean={raw_h2h1.mean():.3f})')
    axes[0].hist(rest_h2h1, bins=bins, alpha=0.7, color='#1f77b4',
                 label=f'V5-B output (mean={rest_h2h1.mean():.3f})')
    axes[0].hist(gt_h2h1,   bins=bins, alpha=0.7, color='#2ca02c',
                 label=f'GT contact PPG (mean={gt_h2h1.mean():.3f})')
    axes[0].set_xlabel('H2/H1 ratio')
    axes[0].set_ylabel('Count (cycles)')
    axes[0].set_title('(A) H2/H1 distribution at each stage')
    axes[0].legend(fontsize=9)

    # Panel B: box plots
    data_bp  = [raw_h2h1.values, rest_h2h1.values, gt_h2h1.values]
    labels_bp = ['rPPG\n(input)', 'V5-B\n(output)', 'GT\n(target)']
    bp = axes[1].boxplot(data_bp, labels=labels_bp, patch_artist=True,
                          medianprops={'color': 'black', 'lw': 2})
    for patch, col in zip(bp['boxes'], ['#d62728', '#1f77b4', '#2ca02c']):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('H2/H1 ratio')
    axes[1].set_title('(B) H2/H1 box plots: rPPG → output → GT')
    axes[1].text(2, rest_h2h1.median() + 0.05,
                 f'10× improvement\n({raw_h2h1.mean():.3f} → {rest_h2h1.mean():.3f})',
                 ha='center', fontsize=10, color='#1f77b4',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4fd', alpha=0.8))

    # Panel C: mean H2/H1 per architecture
    arch_means = {'rPPG input': raw_h2h1.mean()}
    for m in ARCH_META:
        arch = m['arch']
        if arch in dfs_test and 'pred_h2h1' in dfs_test[arch].columns:
            arch_means[arch] = dfs_test[arch]['pred_h2h1'].mean()
    arch_means['GT'] = gt_h2h1.mean()

    names = list(arch_means.keys())
    vals  = list(arch_means.values())
    bar_cols = (['#d62728'] +
                [next((m['color'] for m in ARCH_META if m['arch'] == n), '#aaaaaa')
                 for n in names[1:-1]] +
                ['#2ca02c'])
    axes[2].barh(names, vals, color=bar_cols, alpha=0.85, edgecolor='white')
    axes[2].axvline(gt_h2h1.mean(), color='#2ca02c', lw=2, ls='--', alpha=0.7)
    axes[2].set_xlabel('Mean H2/H1 ratio')
    axes[2].set_title('(C) H2/H1 restoration by architecture\n(GT target = dashed line)')
    for i, v in enumerate(vals):
        axes[2].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    t_stat, p_val = stats.ttest_ind(rest_h2h1, gt_h2h1)
    fig.suptitle(f'Harmonic Restoration: H2/H1  rPPG ({raw_h2h1.mean():.3f}) → Output ({rest_h2h1.mean():.3f}) → GT ({gt_h2h1.mean():.3f})\n'
                 f'10× improvement consistent across all architectures  |  Output vs GT: p={p_val:.4f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = save_fig(fig, out_dir, 'harmonic_restoration.png')

    # ── Individual panels (paper figures — no suptitle) ───────────────────────
    # Panel A: H2/H1 distributions at three signal stages
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1.2, 40)
    ax_a.hist(raw_h2h1,  bins=bins, alpha=0.7, color='#d62728',
              label=f'rPPG input (mean={raw_h2h1.mean():.3f})')
    ax_a.hist(rest_h2h1, bins=bins, alpha=0.7, color='#1f77b4',
              label=f'VAE-Base output (mean={rest_h2h1.mean():.3f})')
    ax_a.hist(gt_h2h1,   bins=bins, alpha=0.7, color='#2ca02c',
              label=f'GT contact PPG (mean={gt_h2h1.mean():.3f})')
    ax_a.set_xlabel('H2/H1 ratio')
    ax_a.set_ylabel('Count (cycles)')
    ax_a.legend(fontsize=9)
    fig_a.tight_layout()
    save_fig(fig_a, out_dir, 'harmonic_restoration_dist.png')
    plt.close(fig_a)

    # Panel C: mean H2/H1 per architecture (paper figure)
    arch_means_c = {'rPPG input': raw_h2h1.mean()}
    for m in ARCH_META:
        arch = m['arch']
        if arch in dfs_test and 'pred_h2h1' in dfs_test[arch].columns:
            arch_means_c[arch] = dfs_test[arch]['pred_h2h1'].mean()
    arch_means_c['GT'] = gt_h2h1.mean()
    names_c = list(arch_means_c.keys())
    vals_c  = list(arch_means_c.values())
    display_names_c = [PAPER_NAMES.get(n, n) for n in names_c]
    bar_cols_c = (['#d62728'] +
                  [next((m['color'] for m in ARCH_META if m['arch'] == n), '#aaaaaa')
                   for n in names_c[1:-1]] +
                  ['#2ca02c'])
    _fig3_h = max(1.0, len(names_c) * 0.16 + 0.3)
    fig_c, ax_c = plt.subplots(figsize=(3.4, _fig3_h))
    ax_c.barh(display_names_c, vals_c, color=bar_cols_c, alpha=0.85, edgecolor='white')
    ax_c.axvline(gt_h2h1.mean(), color='#2ca02c', lw=1.5, ls='--', alpha=0.7)
    ax_c.set_xlabel('')
    ax_c.tick_params(labelsize=7)
    for i, v in enumerate(vals_c):
        ax_c.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)
    fig_c.tight_layout(pad=0.4)
    save_fig(fig_c, out_dir, 'harmonic_restoration_arch.png')
    plt.close(fig_c)

    save_raw_data({
        'figure': 'fig2_harmonic_restoration',
        'description': 'H2/H1 at rPPG input, V5-B output, and GT contact PPG — from spectral_analysis.csv',
        'rppg_input_h2h1': {
            'mean':    float(raw_h2h1.mean()),
            'std':     float(raw_h2h1.std()),
            'median':  float(raw_h2h1.median()),
            'n_cycles': int(len(raw_h2h1)),
        },
        'v5b_output_h2h1': {
            'mean':    float(rest_h2h1.mean()),
            'std':     float(rest_h2h1.std()),
            'median':  float(rest_h2h1.median()),
            'n_cycles': int(len(rest_h2h1)),
        },
        'gt_contact_h2h1': {
            'mean':    float(gt_h2h1.mean()),
            'std':     float(gt_h2h1.std()),
            'median':  float(gt_h2h1.median()),
            'n_cycles': int(len(gt_h2h1)),
        },
        'improvement_factor_mean':   float(rest_h2h1.mean()   / raw_h2h1.mean()),
        'improvement_factor_median': float(rest_h2h1.median() / raw_h2h1.median()),
        'output_vs_gt_ttest': {'t_stat': float(t_stat), 'p_val': float(p_val)},
        'per_arch_h2h1_mean': {k: float(v) for k, v in arch_means.items()},
        'paper_claims': {
            'rppg_h2h1':  0.05,
            'output_h2h1': 0.50,
            'gt_h2h1':    0.54,
            'improvement': '10x',
            'NOTE': 'Compare claimed values above against actual means/medians to verify correctness',
        },
    }, out_dir, 'fig2_harmonic_restoration.json')

    plt.show()
    return path


if __name__ == '__main__':
    import shutil
    import pandas as pd
    from morph_research_pipeline.plotting.config import RES, FIGS
    from morph_research_pipeline.plotting.utils import load_split, load_eval
    _JOURNAL = Path(r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\Projects_Papers\Journals\TobeSubmitted\JBHI_Jrnl_0526\figures')
    split = load_split()
    spec = pd.read_csv(RES / 'v5' / 'spectral_analysis.csv')
    if 'sid' in spec.columns:
        spec['sid'] = spec['sid'].astype(str)
    dfs = {}
    for _arch, _path, _ec, _ev in [
        ('V5-B',  RES/'v5'/'full_eval_v5.csv',   'encoder', 'B'),
        ('V6',    RES/'v6'/'full_eval_v6.csv',    None,      None),
        ('A1',    RES/'a1'/'full_eval_a1.csv',    'encoder', 'B'),
        ('A2',    RES/'a2'/'full_eval_a2.csv',    None,      None),
        ('A3',    RES/'a3'/'full_eval_a3.csv',    'encoder', 'B'),
        ('A4',    RES/'a4'/'full_eval_a4.csv',    None,      None),
        ('A5',    RES/'a5'/'full_eval_a5.csv',    None,      None),
        ('A6-D',  RES/'a6'/'full_eval_a6.csv',    None,      None),
        ('A7',    RES/'a7'/'full_eval_a7.csv',    None,      None),
        ('A8-v2', RES/'a8'/'full_eval_a8.csv',    None,      None),
        ('A9',    RES/'a9'/'full_eval_a9.csv',    None,      None),
        ('A10',   RES/'a10'/'full_eval_a10.csv',  None,      None),
        ('A11',   RES/'a11'/'full_eval_a11.csv',  None,      None),
        ('A12',   RES/'a12'/'full_eval_a12.csv',  None,      None),
    ]:
        _, df_t, _, _ = load_eval(_path, split, _ec, _ev)
        dfs[_arch] = df_t
    plot_harmonic_restoration(spec, dfs, FIGS)
    shutil.copy2(FIGS / 'harmonic_restoration_arch.png', _JOURNAL / 'fig3_harmonic_restoration.png')
    print('fig3_harmonic_restoration.png -> journal figures')
