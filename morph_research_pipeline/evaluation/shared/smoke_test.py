import sys
sys.path.insert(0, r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\rPPG_Morphology_Restore\morph_research_pipeline')

from morph_config import CYCLES_DIR, VAE_CKPT, CKPT_DIR
from pathlib import Path
import numpy as np
import torch

try:
    cycles = list(Path(CYCLES_DIR).rglob('*_cycles.npz'))
    print(f'Cycle files found: {len(cycles)}', flush=True)
    data = np.load(cycles[0])
    print(f'First file: {cycles[0].name}, sid={int(data["sid"])}, gt shape={data["gt_cycles"].shape}', flush=True)

    from models.vae import PPGVAE
    from morph_config import LATENT_DIM
    vae = PPGVAE(latent_dim=LATENT_DIM)
    state = torch.load(VAE_CKPT, map_location='cpu', weights_only=True)
    vae.load_state_dict(state)
    print(f'VAE loaded from {VAE_CKPT}', flush=True)

    from models.encoder import CameraEncoder
    from morph_config import ENCODER_CKPT
    enc = CameraEncoder(latent_dim=LATENT_DIM, in_channels=1, morpho_aux=True)
    enc_path = ENCODER_CKPT['A']
    enc_state = torch.load(enc_path, map_location='cpu', weights_only=True)
    enc.load_state_dict(enc_state['encoder'])
    print(f'Encoder A loaded from {enc_path}', flush=True)

    from models.encoder_v6 import MacroEncoder
    from morph_config import V6_MACRO_CKPT
    macro = MacroEncoder(latent_dim=LATENT_DIM, in_channels=1, num_subjects=374)
    macro.load_state_dict(torch.load(V6_MACRO_CKPT, map_location='cpu', weights_only=True))
    print(f'V6 Macro loaded from {V6_MACRO_CKPT} (num_subjects=374, old checkpoint)', flush=True)
    print('NOTE: V6 needs retraining on corrected 159-subject split', flush=True)

    print('\nALL SMOKE TESTS PASSED', flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'\nFAILED: {e}', flush=True)
