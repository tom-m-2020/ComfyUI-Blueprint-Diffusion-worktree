"""Decode Phase 27 semantic qualification latents without rerunning inference."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")), str(ROOT / "experiments")]
import flux2_coarse_global_local_falsification as phase2

output = ROOT / "experiments" / "terminal_resampling_production_semantic_results"
vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
for name in ("BRIDGE_TRAIN", "ASTRONAUT"):
    latent = torch.load(output / f"{name}_production.pt", map_location="cpu", weights_only=True)
    phase2.save_pixels(vae.decode(latent).cpu(), output / f"{name}_production.png")
