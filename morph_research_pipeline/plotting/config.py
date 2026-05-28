from pathlib import Path
import matplotlib.pyplot as plt

# ── Paths — derived from config/paths.py (set ROOT_E there) ───────────────
try:
    from morph_research_pipeline.config.paths import ROOT_E, DATA_DIR, RESULTS_DIR, FIGS, SPLIT_FILE
except ImportError:
    from ..config.paths import ROOT_E, DATA_DIR, RESULTS_DIR, FIGS, SPLIT_FILE

ROOT = ROOT_E
RES  = RESULTS_DIR
DATA = DATA_DIR

# ── Dataset display names ──────────────────────────────────────────────────
DATASET_NAMES = {
    'ubfc':       'UBFC',
    'stress2023': 'In-house DS1',
    'fps2023':    'In-house DS2',
    'centan':     'In-house DS3',
}

DATASET_COLORS = {
    'ubfc':       '#2166ac',
    'stress2023': '#92c5de',
    'fps2023':    '#4393c3',
    'centan':     '#d1e5f0',
}

DATASET_INFO = {
    'ubfc':       {'sensor': 'CMS50E (fingertip)', 'hz': '64 Hz',    'n': 98},
    'stress2023': {'sensor': 'Polymate (finger)',   'hz': '500 Hz',  'n': 29},
    'fps2023':    {'sensor': 'Polymate (finger)',   'hz': '500 Hz',  'n': 17},
    'centan':     {'sensor': 'Polymate (finger)',   'hz': '1000 Hz', 'n':  9},
}

# ── Architecture registry (all values from documented experimental log) ────
ARCH_META = [
    {'arch': 'Mean baseline', 'family': 'Baseline',       'per_subj_r': 0.770, 'cross_subj_r': None,   'h2h1_err': None,  'color': '#aaaaaa'},
    {'arch': 'GT ceiling',    'family': 'Baseline',       'per_subj_r': None,  'cross_subj_r': 0.601,  'h2h1_err': 0.000, 'color': '#2ca02c'},
    {'arch': 'V5-B',          'family': 'Reconstruction', 'per_subj_r': 0.652, 'cross_subj_r': 0.808,  'h2h1_err': 0.156, 'color': '#1f77b4'},
    {'arch': 'V6',            'family': 'Reconstruction', 'per_subj_r': 0.681, 'cross_subj_r': 0.970,  'h2h1_err': 0.145, 'color': '#4e9dd4'},
    {'arch': 'A1',            'family': 'Reconstruction', 'per_subj_r': 0.716, 'cross_subj_r': 0.9987, 'h2h1_err': 0.132, 'color': '#17becf'},
    {'arch': 'A2',            'family': 'Reconstruction', 'per_subj_r': 0.518, 'cross_subj_r': 0.9929, 'h2h1_err': 0.155, 'color': '#9edae5'},
    {'arch': 'A3',            'family': 'Reconstruction', 'per_subj_r': 0.713, 'cross_subj_r': 0.9972, 'h2h1_err': 0.134, 'color': '#aec7e8'},
    {'arch': 'A4',            'family': 'Reconstruction', 'per_subj_r': 0.549, 'cross_subj_r': 0.9993, 'h2h1_err': 0.149, 'color': '#c5b0d5'},
    {'arch': 'A4-B',          'family': 'Reconstruction', 'per_subj_r': 0.498, 'cross_subj_r': 0.7735, 'h2h1_err': 0.149, 'color': '#9467bd'},
    {'arch': 'A5',            'family': 'Contrastive',    'per_subj_r': 0.656, 'cross_subj_r': 0.960,  'h2h1_err': 0.147, 'color': '#ff7f0e'},
    {'arch': 'A5-v4',         'family': 'Contrastive',    'per_subj_r': 0.656, 'cross_subj_r': 0.892,  'h2h1_err': 0.163, 'color': '#ffbb78'},
    {'arch': 'A6-D',          'family': 'Reconstruction', 'per_subj_r': 0.903, 'cross_subj_r': 0.996,  'h2h1_err': 0.143, 'color': '#d62728'},
    {'arch': 'A7',            'family': 'Reconstruction', 'per_subj_r': 0.850, 'cross_subj_r': 0.9999, 'h2h1_err': 0.152, 'color': '#ff9896'},
    {'arch': 'A8-v2',         'family': 'Reconstruction', 'per_subj_r': 0.644, 'cross_subj_r': 0.9957, 'h2h1_err': 0.161, 'color': '#e377c2'},
    {'arch': 'A9',            'family': 'Diffusion',      'per_subj_r': 0.599, 'cross_subj_r': 0.9947, 'h2h1_err': 0.172, 'color': '#8c564b'},
    {'arch': 'A10',           'family': 'Diffusion',      'per_subj_r': 0.540, 'cross_subj_r': 0.993,  'h2h1_err': 0.178, 'color': '#c49c94'},
    {'arch': 'A11',           'family': 'Contrastive',    'per_subj_r': 0.729, 'cross_subj_r': 0.9995, 'h2h1_err': 0.139, 'color': '#bcbd22'},
    {'arch': 'A12',           'family': 'Contrastive',    'per_subj_r': 0.724, 'cross_subj_r': 0.9999, 'h2h1_err': 0.141, 'color': '#dbdb8d'},
    {'arch': 'A13',           'family': 'Contrastive',    'per_subj_r': None,  'cross_subj_r': None,   'h2h1_err': None,  'color': '#7f7f7f'},
]

ARCH_FAMILY_MARKER = {'Reconstruction': 'o', 'Contrastive': 's', 'Diffusion': '^', 'Baseline': 'D'}

# Paper display names — internal codes stay unchanged everywhere else
PAPER_NAMES = {
    'V5-B':      'VAE-Base',
    'V6':        'VAE-Orth',
    'A1':        'VAE-Large',
    'A2':        'VAE-Flow',
    'A3':        'VQ-VAE',
    'A4':        'Trans-Multi',
    'A4-B':      'Trans-rPPG',
    'A5':        'Two-Stage',
    'A5-v4':     'Two-Stage+Div',
    'A6-D':      'RGB-Window',
    'A7':        'RGB-Physics',
    'A8-v2':     'RGB-FPS',
    'A9':        'Diffusion-z',
    'A10':       'DPS-rPPG',
    'A11':       'VMD-6ch',
    'A12':       'VMD-Peak',
    'A13':       'SupCon',
    'GT ceiling':    'GT ceiling',
    'Mean baseline': 'Mean baseline',
}

# ── Matplotlib style ───────────────────────────────────────────────────────
PLOT_RC = {
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
}


def apply_rc():
    plt.rcParams.update(PLOT_RC)


def ds_label(key: str) -> str:
    return DATASET_NAMES.get(key, key)


def arch_color(arch_name: str) -> str:
    for m in ARCH_META:
        if m['arch'] == arch_name:
            return m['color']
    return '#888888'
