import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
try:
    from .config import apply_rc, ARCH_META, ARCH_FAMILY_MARKER, PAPER_NAMES
    from .utils import save_fig, save_raw_data
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc, ARCH_META, ARCH_FAMILY_MARKER, PAPER_NAMES
    from morph_research_pipeline.plotting.utils import save_fig, save_raw_data

# Display-only x-offsets for near-coincident point pairs (data values unchanged in raw JSON)
_DISPLAY_JITTER = {
    'A1':   (+0.013, 0.0),   # VAE-Large and VQ-VAE nearly coincide (~0.715, ~0.999)
    'A3':   (-0.013, 0.0),
    'A11':  (+0.009, 0.0),   # VMD-6ch and VMD-Peak nearly coincide (~0.726, ~0.9997)
    'A12':  (-0.009, 0.0),
}


def plot_collapse_scatter(out_dir, figsize=(3.4, 3.4)):
    apply_rc()

    X_LO, X_HI = 0.450, 0.950
    Y_LO, Y_HI = 0.550, 1.030

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)

    # ── GT ceiling + target zone ──────────────────────────────────────────
    ax.axhline(0.601, color='#2ca02c', lw=2.0, ls='--', alpha=0.9)
    ax.axvline(0.770, color='#aaaaaa', lw=1.5, ls=':',  alpha=0.8)
    ax.fill_between([X_LO, X_HI], Y_LO, 0.601, alpha=0.08, color='#2ca02c')
    ax.text(X_LO + 0.015, 0.604, 'GT ceiling (0.601)',
            fontsize=6, color='#2ca02c')

    # ── Architecture scatter ──────────────────────────────────────────────
    arch_handles = []
    for m in ARCH_META:
        psr = m['per_subj_r']
        csr = m['cross_subj_r']
        if psr is None or csr is None:
            continue
        dx, dy = _DISPLAY_JITTER.get(m['arch'], (0, 0))
        mk = ARCH_FAMILY_MARKER.get(m['family'], 'o')
        ax.scatter(psr + dx, csr + dy, s=50, marker=mk, color=m['color'],
                   edgecolors='black', linewidths=0.5, zorder=5)
        arch_handles.append(
            plt.Line2D([0], [0], marker=mk, color='w',
                       markerfacecolor=m['color'], markeredgecolor='black',
                       markeredgewidth=0.4, markersize=5,
                       label=PAPER_NAMES.get(m['arch'], m['arch']))
        )

    # ── Annotations ───────────────────────────────────────────────────────
    ax.annotate('Best anti-collapse\n(Trans-rPPG)',
                xy=(0.498, 0.7735), xytext=(0.460, 0.820),
                fontsize=6, color='#9467bd',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.0))
    ax.annotate('Collapse paradox\n(RGB-Window)',
                xy=(0.903, 0.996), xytext=(0.903, 0.925),
                fontsize=6, color='#d62728', ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.0))

    ax.set_xlabel('Per-subject $r$  (accuracy, higher is better)', fontsize=8)
    ax.set_ylabel('Cross-subject $r$  (collapse)', fontsize=8)
    ax.tick_params(labelsize=7)

    # ── Family legend — lower-left inside axes ────────────────────────────
    family_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                   markeredgecolor='black', markeredgewidth=0.5, markersize=5,
                   label='Reconstruction'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
                   markeredgecolor='black', markeredgewidth=0.5, markersize=5,
                   label='Contrastive'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#888888',
                   markeredgecolor='black', markeredgewidth=0.5, markersize=5,
                   label='Diffusion'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#888888',
                   markeredgecolor='black', markeredgewidth=0.5, markersize=5,
                   label='Baseline'),
    ]
    family_leg = ax.legend(handles=family_handles, loc='lower left',
                           fontsize=6, frameon=True, framealpha=0.95,
                           title='Shape = Family', title_fontsize=6,
                           handlelength=0.8, handletextpad=0.3,
                           borderpad=0.5, labelspacing=0.3)
    ax.add_artist(family_leg)

    # ── Architecture legend below axes (4 cols) ───────────────────────────
    ax.legend(handles=arch_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=6,
              frameon=True, framealpha=0.9,
              handlelength=0.8, handletextpad=0.3, columnspacing=0.5,
              borderpad=0.4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.40)
    path = save_fig(fig, out_dir, 'collapse_accuracy_scatter.png')

    save_raw_data({
        'figure': 'fig6_collapse_scatter',
        'description': (
            'Per-subject r (accuracy, higher is better) vs cross-subject r '
            '(collapse, lower is better) for all 13 architectures. '
            'Near-coincident pairs A1/A3 and A11/A12 are jittered ±0.013 and ±0.009 '
            'in x for visibility; actual data values are unchanged.'
        ),
        'gt_ceiling': {'cross_subj_r': 0.601},
        'trivial_baseline': {'per_subj_r': 0.770},
        'architectures': [
            {
                'arch': m['arch'],
                'paper_name': PAPER_NAMES.get(m['arch'], m['arch']),
                'family': m['family'],
                'per_subj_r': m['per_subj_r'],
                'cross_subj_r': m['cross_subj_r'],
            }
            for m in ARCH_META
        ],
        'paper_values': {
            'best_anti_collapse': {
                'arch': 'Trans-rPPG', 'cross_subj_r': 0.7735, 'per_subj_r': 0.498,
            },
            'collapse_paradox': {
                'arch': 'RGB-Window', 'per_subj_r': 0.903, 'cross_subj_r': 0.996,
                'note': 'Highest per-subject accuracy but near-complete collapse',
            },
        },
    }, out_dir, 'fig6_collapse_scatter.json')

    plt.show()
    return path


if __name__ == '__main__':
    import shutil
    from morph_research_pipeline.plotting.config import FIGS
    _JOURNAL = Path(r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\Projects_Papers\Journals\TobeSubmitted\JBHI_Jrnl_0526\figures')
    plot_collapse_scatter(FIGS)
    shutil.copy2(FIGS / 'collapse_accuracy_scatter.png', _JOURNAL / 'fig6_collapse_scatter.png')
    print('fig6_collapse_scatter.png -> journal figures')
