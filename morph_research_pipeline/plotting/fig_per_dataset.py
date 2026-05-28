import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
try:
    from .config import apply_rc, ARCH_META, DATASET_NAMES, ds_label
    from .utils import save_fig, save_raw_data
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc, ARCH_META, DATASET_NAMES, ds_label
    from morph_research_pipeline.plotting.utils import save_fig, save_raw_data


def plot_per_dataset(dfs_test, split_df, out_dir,
                     arch_order=None, figsize=(15, 6)):
    apply_rc()
    if arch_order is None:
        arch_order = ['V5-B', 'V6', 'A1', 'A2', 'A3', 'A4',
                      'A5', 'A5-v4', 'A6-D', 'A7', 'A8-v2',
                      'A9', 'A10', 'A11', 'A12']

    test_split = split_df[split_df['split'] == 'test']
    ds_keys    = list(DATASET_NAMES.keys())
    ds_labels  = [ds_label(k) for k in ds_keys]

    # Build matrix
    matrix = {}
    for arch in arch_order:
        if arch not in dfs_test:
            continue
        df_t = dfs_test[arch]
        row  = {}
        for dk in ds_keys:
            sids = test_split[test_split['dataset'] == dk]['sid'].astype(str)
            df_ds = df_t[df_t['sid'].isin(sids)]
            row[ds_label(dk)] = df_ds.groupby('sid')['shape_r'].mean().mean() if len(df_ds) else np.nan
        matrix[arch] = row

    df_mat = pd.DataFrame(matrix).T
    df_mat.columns = ds_labels

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: heatmap
    sns.heatmap(df_mat.astype(float), ax=axes[0], annot=True, fmt='.3f',
                cmap='RdYlGn', vmin=0.1, vmax=1.0, linewidths=0.5,
                annot_kws={'size': 6},
                cbar_kws={'label': 'Per-subject r'})
    axes[0].set_title('(A) Per-dataset performance heatmap\n(green = high, red = low/collapse)')
    axes[0].set_xlabel('Dataset', fontsize=8)
    axes[0].set_ylabel('Architecture', fontsize=8)
    axes[0].tick_params(labelsize=7)

    # Panel B: UBFC vs In-house DS3 for selected architectures
    sel_archs = ['V5-B', 'V6', 'A5', 'A6-D']
    ubfc_label  = ds_label('ubfc')
    centan_label = ds_label('centan')
    ubfc_vals   = [df_mat.loc[a, ubfc_label]   if a in df_mat.index else np.nan for a in sel_archs]
    centan_vals = [df_mat.loc[a, centan_label] if a in df_mat.index else np.nan for a in sel_archs]

    x2 = np.arange(len(sel_archs))
    axes[1].bar(x2 - 0.2, ubfc_vals,   0.35, color='#1f77b4', alpha=0.85, label='UBFC',           edgecolor='white')
    axes[1].bar(x2 + 0.2, centan_vals, 0.35, color='#d62728', alpha=0.85, label='In-house DS3',   edgecolor='white')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(sel_archs)
    axes[1].set_ylabel('Per-subject r', fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f'(B) UBFC vs {centan_label}\nV6 {centan_label} collapse highlighted')
    axes[1].legend(fontsize=6)
    axes[1].tick_params(labelsize=7)
    axes[1].axhline(0.77, color='gray', lw=1.5, ls=':', alpha=0.7)

    # Annotate V6 In-house DS3 collapse
    if 'V6' in sel_archs:
        v6_idx = sel_archs.index('V6')
        v = centan_vals[v6_idx]
        if not np.isnan(v):
            axes[1].annotate(f'V6 {centan_label}\ncollapse',
                             xy=(v6_idx + 0.2, v), xytext=(v6_idx + 0.6, 0.35),
                             fontsize=6, color='#d62728',
                             arrowprops=dict(arrowstyle='->', color='#d62728'))

    fig.suptitle('Per-Dataset Performance Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = save_fig(fig, out_dir, 'per_dataset_breakdown.png')

    save_raw_data({
        'figure': 'sfig1_per_dataset',
        'description': 'Per-architecture per-dataset mean per-subject r heatmap',
        'datasets': ds_labels,
        'architectures': list(df_mat.index),
        'per_arch_per_dataset_r': {
            arch: {
                ds: (float(v) if not (isinstance(v, float) and np.isnan(v)) else None)
                for ds, v in row.items()
            }
            for arch, row in df_mat.iterrows()
        },
    }, out_dir, 'sfig1_per_dataset.json')

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
    plot_per_dataset(dfs, split, FIGS)
    shutil.copy2(FIGS / 'per_dataset_breakdown.png', _JOURNAL / 'sfig1_per_dataset.png')
    print('sfig1_per_dataset.png -> journal figures')
