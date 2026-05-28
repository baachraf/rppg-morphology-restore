import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path
try:
    from .config import apply_rc
    from .utils import save_fig
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc
    from morph_research_pipeline.plotting.utils import save_fig


def plot_gt_diversity(df_v5_test, out_dir, figsize=(3.4, 4.0)):
    """GT morphological diversity — single-column, 2 panels stacked."""
    apply_rc()

    gt = df_v5_test.groupby('sid').agg(
        gt_h2h1_mean=('gt_h2h1',   'mean'),
        pred_h2h1_mean=('pred_h2h1', 'mean'),
        gt_ipa_mean=('gt_ipa',     'mean'),
        pred_ipa_mean=('pred_ipa',  'mean'),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Panel 1 — GT H2/H1 distribution
    axes[0].hist(gt['gt_h2h1_mean'], bins=15, color='#2ca02c', alpha=0.75, edgecolor='white')
    axes[0].axvline(gt['gt_h2h1_mean'].mean(), color='darkgreen', lw=1.5, ls='--',
                    label=f'$\mu = {gt["gt_h2h1_mean"].mean():.3f}$')
    axes[0].set_ylabel('Count', fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].legend(fontsize=7)

    # Panel 2 — GT vs predicted scatter
    r_val, _ = stats.pearsonr(gt['gt_h2h1_mean'], gt['pred_h2h1_mean'])
    pad = 0.05
    x_lo = gt['gt_h2h1_mean'].min() - pad
    x_hi = gt['gt_h2h1_mean'].max() + pad
    y_lo = gt['pred_h2h1_mean'].min() - pad
    y_hi = gt['pred_h2h1_mean'].max() + pad
    axes[1].scatter(gt['gt_h2h1_mean'], gt['pred_h2h1_mean'],
                    alpha=0.7, s=20, color='#1f77b4', edgecolors='white', linewidths=0.3)
    lim = [min(x_lo, y_lo), max(x_hi, y_hi)]
    axes[1].plot(lim, lim, 'k--', alpha=0.3, lw=1.0)
    axes[1].set_xlim(x_lo, x_hi)
    axes[1].set_ylim(y_lo, y_hi)
    axes[1].set_ylabel('Pred. H2/H1', fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].text(0.05, 0.88, f'$r = {r_val:.3f}$', transform=axes[1].transAxes, fontsize=7)

    plt.tight_layout(h_pad=0.8)
    path = save_fig(fig, out_dir, 'gt_diversity.png')
    return path


if __name__ == '__main__':
    import shutil
    from morph_research_pipeline.plotting.config import RES, FIGS
    from morph_research_pipeline.plotting.utils import load_split, load_eval
    _JOURNAL = Path(r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\Projects_Papers\Journals\TobeSubmitted\JBHI_Jrnl_0526\figures')
    split = load_split()
    _, df_v5_test, _, _ = load_eval(RES / 'v5' / 'full_eval_v5.csv', split, 'encoder', 'B')
    plot_gt_diversity(df_v5_test, FIGS)
    shutil.copy2(FIGS / 'gt_diversity.png', _JOURNAL / 'fig1_gt_diversity.png')
    print('fig1_gt_diversity.png -> journal figures')
