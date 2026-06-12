"""Central configuration & path resolution for portability.

Every machine-specific location resolves from the repo root or an environment
variable, so the project runs unchanged on any machine (dev box, H100, or a
grader reproducing from GitHub). Nothing here is hardcoded to a user's home dir.

Overrides (all optional):
  IRIS_ROREM_CKPT, IRIS_SDXL_BASE, IRIS_TRELLIS_DIR
  IRIS_TRELLIS_PYTHON   (trellis conda env interpreter)
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(env, *default_parts):
    return os.environ.get(env) or os.path.join(REPO_ROOT, *default_parts)


# ── third-party model repo (cloned into models/, see scripts/setup_envs.sh) ──
TRELLIS_DIR = _path("IRIS_TRELLIS_DIR", "models", "TRELLIS")

# ── checkpoints ──
# RORem UNet weights (download via scripts/fetch_weights.py or set the env var)
ROREM_CKPT = _path("IRIS_ROREM_CKPT", "checkpoints", "RORem")
SDXL_BASE = os.environ.get("IRIS_SDXL_BASE", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")

# HF model ids (open-weight; auto-download on first use)
VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"
SAM3_ID = "facebook/sam3"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Large-hf"
TRELLIS_ID = "microsoft/TRELLIS-image-large"
VGGT_ID = "facebook/VGGT-1B"
MASK2FORMER_ID = "facebook/mask2former-swin-large-ade-semantic"


def conda_env_python(env_name):
    """Resolve a named conda env's python interpreter, portably.

    Order: IRIS_<ENV>_PYTHON override → CONDA_EXE root → common install roots.
    Used to launch the TRELLIS subprocess worker, which lives in
    their own envs (dependency conflicts, not memory)."""
    override = os.environ.get(f"IRIS_{env_name.upper()}_PYTHON")
    if override:
        return override
    candidates = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        root = os.path.dirname(os.path.dirname(conda_exe))
        candidates.append(os.path.join(root, "envs", env_name, "bin", "python"))
    for root in ("~/anaconda3", "~/miniconda3", "~/miniforge3", "~/mambaforge", "/opt/conda"):
        candidates.append(os.path.join(os.path.expanduser(root), "envs", env_name, "bin", "python"))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError(
        f"conda env '{env_name}' python not found. Set IRIS_{env_name.upper()}_PYTHON "
        f"to its interpreter path, or create the env (scripts/setup_envs.sh). Tried: {candidates}")
