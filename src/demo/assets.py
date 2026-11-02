"""Fetches the private asset repository (model weights, input-check statistics, sample images, Grad-CAM colour scale) next to the code.

Used by the Streamlit app. The assets are not public: the token comes from the caller (a Streamlit secret or an environment variable)
and is never printed or logged. The same files are listed in deploy/space/fetch_assets.py for the Docker Space.
"""
from __future__ import annotations

from pathlib import Path

NEEDED = ["artifacts/deploy/gate_params.json", "artifacts/deploy/model/CURRENT", "artifacts/torch_cache/hub/checkpoints/resnet18-f37072fd.pth",
          "artifacts/demo_pack/examples.json", "data/metadata/manifest.csv", "reports/11_gradcam_classic.json"]


def missing(root: Path) -> list[str]:
    """Files the demo needs that are not under `root` yet."""
    gone = [rel for rel in NEEDED if not (root / rel).exists()]
    if not list((root / "artifacts/deploy/model").glob("*/member_*.pt")):
        gone.append("artifacts/deploy/model/*/member_*.pt")
    return gone


def ensure_assets(root: Path, repo: str | None, token: str | None) -> list[str]:
    """Download the private repo into `root` unless everything is there already. Returns what is still missing (empty = ready).
    The repository's own README is not copied, so a README next to the code is never overwritten."""
    gone = missing(root)
    if not gone or not repo or not token:
        return gone
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(root), token=token, ignore_patterns=["README.md", ".gitattributes"])
    return missing(root)
