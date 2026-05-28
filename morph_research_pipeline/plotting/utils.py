import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from .config import SPLIT_FILE, DATASET_NAMES


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return {str(k): (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def save_raw_data(data, out_dir, filename):
    """Save figure raw data as JSON to <out_dir>/raw_data/<filename>."""
    raw_dir = Path(out_dir) / 'raw_data'
    raw_dir.mkdir(parents=True, exist_ok=True)
    data['_generated_at'] = datetime.now().isoformat(timespec='seconds')
    p = raw_dir / filename
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f'Raw data saved: {p}')
    return p


def load_split(split_file=None):
    f = Path(split_file) if split_file else SPLIT_FILE
    df = pd.read_csv(f)
    df['sid'] = df['sid'].astype(str)
    return df


def load_eval(path, split_df, encoder_col=None, encoder_val=None, r_col='shape_r'):
    """Load eval CSV, optionally filter by encoder, restrict to test set.

    Returns (df_all, df_test, mean_per_subj_r, per_subj_series).
    """
    df = pd.read_csv(path)
    df['sid'] = df['sid'].astype(str)
    if encoder_col and encoder_val:
        df = df[df[encoder_col] == encoder_val]
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])
    df_test = df[df['sid'].isin(test_sids)].copy()
    per_subj = df_test.groupby('sid')[r_col].mean()
    return df, df_test, per_subj.mean(), per_subj


def per_dataset_r(df_test, r_col='shape_r'):
    """Per-subject r broken down by dataset; index uses display names."""
    raw = df_test.groupby(['dataset', 'sid'])[r_col].mean().groupby('dataset').mean()
    return raw.rename(index=DATASET_NAMES)


def morpho_stats(df_test):
    h2h1_err      = df_test['h2h1_error'].abs().mean()              if 'h2h1_error' in df_test.columns else np.nan
    ipa_err       = df_test['ipa_error'].abs().mean()               if 'ipa_error'   in df_test.columns else np.nan
    pred_h2h1_std = df_test.groupby('sid')['pred_h2h1'].mean().std() if 'pred_h2h1'  in df_test.columns else np.nan
    gt_h2h1_std   = df_test.groupby('sid')['gt_h2h1'].mean().std()   if 'gt_h2h1'    in df_test.columns else np.nan
    return {'h2h1_err': h2h1_err, 'ipa_err': ipa_err,
            'pred_h2h1_std': pred_h2h1_std, 'gt_h2h1_std': gt_h2h1_std}


def save_fig(fig, out_dir, filename, dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / filename
    fig.savefig(str(p), dpi=dpi, bbox_inches='tight')
    print(f'Saved: {p}')
    return p
