import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
try:
    from .config import apply_rc, DATASET_NAMES, ds_label
    from .utils import save_fig, save_raw_data
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc, DATASET_NAMES, ds_label
    from morph_research_pipeline.plotting.utils import save_fig, save_raw_data


def plot_zero_shot(df_v5_all_B, split_df, out_dir, figsize=(14, 5)):
    apply_rc()
    test_split = split_df[split_df['split'] == 'test']

    ubfc_sids     = test_split[test_split['dataset'] == 'ubfc']['sid'].astype(str)
    internal_sids = test_split[test_split['dataset'] != 'ubfc']['sid'].astype(str)

    df_ubfc = df_v5_all_B[df_v5_all_B['sid'].isin(ubfc_sids)]
    df_int  = df_v5_all_B[df_v5_all_B['sid'].isin(internal_sids)]

    ubfc_persubj = df_ubfc.groupby('sid')['shape_r'].mean()
    int_persubj  = df_int.groupby('sid')['shape_r'].mean()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: UBFC distribution
    bins = np.linspace(0.2, 1.0, 20)
    axes[0].hist(ubfc_persubj, bins=bins, color='#1f77b4', alpha=0.8,
                 label=f'UBFC (n={len(ubfc_persubj)})')
    axes[0].axvline(ubfc_persubj.mean(), color='#1f77b4', lw=2.5, ls='--',
                    label=f'UBFC mean = {ubfc_persubj.mean():.3f}')
    axes[0].axvline(0.770, color='gray', lw=1.5, ls=':', label='Trivial baseline (0.77)')
    axes[0].set_xlabel('Per-subject r', fontsize=8)
    axes[0].set_ylabel('Count', fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].set_title('(A) UBFC zero-shot per-subject r\n(V5-B, unseen CMS50E 64 Hz reference)')
    axes[0].legend(fontsize=6)

    # Panel B: UBFC vs internal box
    bp = axes[1].boxplot([ubfc_persubj.values, int_persubj.values],
                          labels=['UBFC\n(zero-shot)', 'In-house\n(seen domain)'],
                          patch_artist=True, medianprops={'color': 'black', 'lw': 2})
    bp['boxes'][0].set_facecolor('#1f77b4'); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#2ca02c'); bp['boxes'][1].set_alpha(0.7)
    axes[1].axhline(0.77, color='gray', lw=1.5, ls=':', alpha=0.7)
    axes[1].set_ylabel('Per-subject r', fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].set_title('(B) Zero-shot UBFC vs in-house test\n(generalisation without retraining)')

    # Panel C: per-dataset r
    ds_r = {}
    for ds_key, ds_name in DATASET_NAMES.items():
        ds_sids = test_split[test_split['dataset'] == ds_key]['sid'].astype(str)
        df_ds   = df_v5_all_B[df_v5_all_B['sid'].isin(ds_sids)]
        if len(df_ds):
            ds_r[ds_name] = df_ds.groupby('sid')['shape_r'].mean().mean()

    ds_colors_map = {
        DATASET_NAMES['ubfc']:       '#1f77b4',
        DATASET_NAMES['stress2023']: '#ff7f0e',
        DATASET_NAMES['fps2023']:    '#2ca02c',
        DATASET_NAMES['centan']:     '#d62728',
    }
    bars = axes[2].bar(list(ds_r.keys()), list(ds_r.values()),
                        color=[ds_colors_map.get(d, '#999') for d in ds_r],
                        alpha=0.85, edgecolor='white')
    axes[2].axhline(0.77, color='gray', lw=1.5, ls=':', alpha=0.7, label='Trivial baseline')
    axes[2].set_ylabel('Per-subject r', fontsize=8)
    axes[2].set_title('(C) V5-B performance by dataset\n(In-house DS3 collapse noted in V6)')
    axes[2].set_ylim(0, 1.05)
    axes[2].tick_params(axis='x', rotation=10, labelsize=7)
    axes[2].tick_params(axis='y', labelsize=7)
    for bar in bars:
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{bar.get_height():.3f}', ha='center', fontsize=6)
    axes[2].legend(fontsize=6)

    fig.suptitle('Zero-Shot Generalisation: V5-B on UBFC Without Retraining\n'
                 'Clinical (500 Hz Polymate) → Consumer (30 fps CMS50E)  |  UBFC r = '
                 f'{ubfc_persubj.mean():.3f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = save_fig(fig, out_dir, 'zero_shot_ubfc.png')

    save_raw_data({
        'figure': 'sfig2_zero_shot',
        'description': 'V5-B zero-shot to UBFC-PHYS (CMS50E 64 Hz, never seen during training)',
        'ubfc': {
            'n_subjects':      int(len(ubfc_persubj)),
            'mean_per_subj_r': float(ubfc_persubj.mean()),
            'std_per_subj_r':  float(ubfc_persubj.std()),
            'per_subject_r':   {str(k): float(v) for k, v in ubfc_persubj.items()},
        },
        'internal': {
            'n_subjects':      int(len(int_persubj)),
            'mean_per_subj_r': float(int_persubj.mean()),
            'std_per_subj_r':  float(int_persubj.std()),
            'per_subject_r':   {str(k): float(v) for k, v in int_persubj.items()},
        },
        'per_dataset_r': {k: float(v) for k, v in ds_r.items()},
        'trivial_baseline_per_subj_r': 0.770,
        'paper_values': {
            'reported_ubfc_mean_r': 0.751,
        },
    }, out_dir, 'sfig2_zero_shot.json')

    plt.show()
    return path


if __name__ == '__main__':
    import shutil
    from morph_research_pipeline.plotting.config import RES, FIGS
    from morph_research_pipeline.plotting.utils import load_split, load_eval
    _JOURNAL = Path(r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\Projects_Papers\Journals\TobeSubmitted\JBHI_Jrnl_0526\figures')
    split = load_split()
    df_v5_all_B, _, _, _ = load_eval(RES / 'v5' / 'full_eval_v5.csv', split, 'encoder', 'B')
    plot_zero_shot(df_v5_all_B, split, FIGS)
    shutil.copy2(FIGS / 'zero_shot_ubfc.png', _JOURNAL / 'sfig2_zero_shot.png')
    print('sfig2_zero_shot.png -> journal figures')
