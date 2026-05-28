import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
try:
    from .config import apply_rc, ARCH_META, PAPER_NAMES
    from .utils import save_fig, save_raw_data
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc, ARCH_META, PAPER_NAMES
    from morph_research_pipeline.plotting.utils import save_fig, save_raw_data


def plot_template_collapse(dfs_test, out_dir, highlight=('V5-B', 'A5', 'A5-v4', 'A6-D'),
                           figsize=(15, 5)):
    apply_rc()

    # Collect per-subject H2/H1 stats
    arch_h2h1 = {}
    for m in ARCH_META:
        arch = m['arch']
        if arch not in dfs_test:
            continue
        df_t = dfs_test[arch]
        if 'pred_h2h1' not in df_t.columns or 'gt_h2h1' not in df_t.columns:
            continue
        arch_h2h1[arch] = df_t.groupby('sid')[['pred_h2h1', 'gt_h2h1']].mean()

    if not arch_h2h1:
        print('No H2/H1 columns found in any architecture.')
        return None

    archs_with_data = list(arch_h2h1.keys())
    gt_stds   = [arch_h2h1[a]['gt_h2h1'].std()   for a in archs_with_data]
    pred_stds = [arch_h2h1[a]['pred_h2h1'].std() for a in archs_with_data]
    colors    = [next(m['color'] for m in ARCH_META if m['arch'] == a) for a in archs_with_data]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: inter-subject std of predicted H2/H1
    x = np.arange(len(archs_with_data))
    bars = axes[0].bar(x, pred_stds, color=colors, alpha=0.85, edgecolor='white')
    axes[0].axhline(np.mean(gt_stds), color='#2ca02c', lw=2.5, ls='--',
                    label=f'GT H2/H1 std ≈ {np.mean(gt_stds):.3f}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PAPER_NAMES.get(a, a) for a in archs_with_data], rotation=35, ha='right')
    axes[0].set_ylabel('Std of per-subject mean predicted H2/H1')
    axes[0].set_title('(A) Inter-subject diversity in predicted H2/H1\n'
                      'Collapsed model ≈ low std.  Subject-specific ≈ std near GT level')
    axes[0].legend()
    for bar, v in zip(bars, pred_stds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # Panel B: GT vs predicted H2/H1 scatter for selected architectures
    markers = ['o', 's', '^', 'D']
    for arch, mk in zip(highlight, markers):
        if arch not in arch_h2h1:
            continue
        by_subj = arch_h2h1[arch]
        col = next(m['color'] for m in ARCH_META if m['arch'] == arch)
        csr = next((m['cross_subj_r'] for m in ARCH_META if m['arch'] == arch), None)
        label = f'{PAPER_NAMES.get(arch, arch)} (xr={csr:.3f})' if csr else PAPER_NAMES.get(arch, arch)
        axes[1].scatter(by_subj['gt_h2h1'], by_subj['pred_h2h1'],
                        s=60, marker=mk, color=col, alpha=0.75,
                        edgecolors='white', linewidths=0.5, label=label)

    lims = [0.15, 0.90]
    axes[1].plot(lims, lims, 'k--', alpha=0.3, lw=1.5, label='y = x (perfect recovery)')
    axes[1].set_xlabel('GT H2/H1 (per subject)')
    axes[1].set_ylabel('Predicted H2/H1 (per subject)')
    axes[1].set_title('(B) GT vs predicted H2/H1 per subject\n'
                      'Collapsed model → flat horizontal cluster.  Subject-specific → diagonal')
    axes[1].legend(fontsize=9)

    fig.suptitle('Template Collapse Quantification via H2/H1 Inter-Subject Diversity',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = save_fig(fig, out_dir, 'template_collapse_visualisation.png')

    # ── Individual panels (paper figures — no suptitle) ───────────────────────
    # Panel A: inter-subject std — horizontal bar chart (names on y-axis, no overlap)
    _FS_LABEL  = 7
    _FS_TICK   = 7
    _FS_LEGEND = 6
    _FS_ANNOT  = 7

    names_a = [PAPER_NAMES.get(a, a) for a in archs_with_data]
    _fig4_h = max(1.0, len(names_a) * 0.16 + 0.3)
    fig_a, ax_a = plt.subplots(figsize=(3.4, _fig4_h))
    bars_a = ax_a.barh(names_a, pred_stds, color=colors, alpha=0.85, edgecolor='white')
    ax_a.axvline(np.mean(gt_stds), color='#2ca02c', lw=2.0, ls='--')
    ax_a.set_xlabel('')
    ax_a.tick_params(axis='both', labelsize=_FS_TICK)
    x_max = np.mean(gt_stds) * 1.8
    ax_a.set_xlim(0, x_max)
    for bar, v in zip(bars_a, pred_stds):
        ax_a.text(v + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                  f'{v:.3f}', ha='left', va='center', fontsize=_FS_ANNOT)
    fig_a.tight_layout(pad=0.4)
    save_fig(fig_a, out_dir, 'template_collapse_std.png')
    plt.close(fig_a)

    # Panel B: GT vs predicted H2/H1 scatter per subject
    markers = ['o', 's', '^', 'D']
    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    for arch, mk in zip(highlight, markers):
        if arch not in arch_h2h1:
            continue
        by_subj = arch_h2h1[arch]
        col = next(m['color'] for m in ARCH_META if m['arch'] == arch)
        csr = next((m['cross_subj_r'] for m in ARCH_META if m['arch'] == arch), None)
        label = f'{PAPER_NAMES.get(arch, arch)} (xr={csr:.3f})' if csr else PAPER_NAMES.get(arch, arch)
        ax_b.scatter(by_subj['gt_h2h1'], by_subj['pred_h2h1'],
                     s=60, marker=mk, color=col, alpha=0.75,
                     edgecolors='white', linewidths=0.5, label=label)
    lims = [0.15, 0.90]
    ax_b.plot(lims, lims, 'k--', alpha=0.3, lw=1.5, label='y = x (perfect recovery)')
    ax_b.set_xlabel('GT H2/H1 (per subject)')
    ax_b.set_ylabel('Predicted H2/H1 (per subject)')
    ax_b.legend(fontsize=9)
    fig_b.tight_layout()
    save_fig(fig_b, out_dir, 'template_collapse_scatter.png')
    plt.close(fig_b)

    per_subject_highlight = {}
    for arch in highlight:
        if arch in arch_h2h1:
            by_subj = arch_h2h1[arch]
            per_subject_highlight[arch] = {
                'gt_h2h1':   by_subj['gt_h2h1'].to_dict(),
                'pred_h2h1': by_subj['pred_h2h1'].to_dict(),
            }

    save_raw_data({
        'figure': 'fig5_template_collapse',
        'description': 'Inter-subject H2/H1 diversity: predicted std vs GT std per architecture',
        'gt_h2h1_std_mean': float(np.mean(gt_stds)) if gt_stds else None,
        'per_arch_pred_h2h1_std': {
            a: float(s) for a, s in zip(archs_with_data, pred_stds)
        },
        'per_arch_gt_h2h1_std': {
            a: float(s) for a, s in zip(archs_with_data, gt_stds)
        },
        'highlighted_per_subject': per_subject_highlight,
    }, out_dir, 'fig5_template_collapse.json')

    plt.show()
    return path


if __name__ == '__main__':
    import shutil
    from morph_research_pipeline.plotting.config import RES, FIGS
    from morph_research_pipeline.plotting.utils import load_split, load_eval
    _JOURNAL = Path(r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\Projects_Papers\Journals\TobeSubmitted\JBHI_Jrnl_0526\figures')
    split = load_split()
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
    plot_template_collapse(dfs, FIGS)
    shutil.copy2(FIGS / 'template_collapse_std.png', _JOURNAL / 'fig4_template_collapse.png')
    print('fig4_template_collapse.png -> journal figures')
