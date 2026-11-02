#!/usr/bin/env python3
"""Downloads the private asset repo (model weights, input-check statistics, sample images, Grad-CAM colour scale) next to the code.

The assets are not public. The token comes from the environment variable HF_TOKEN (a Space secret at run time) or, during the
Docker build, from the secret file /run/secrets/HF_TOKEN. The token is never printed. Exit status 0 when everything the demo
needs is in place, 1 otherwise.

  PTIC_ASSETS_REPO   id of the private repo, e.g. "<user>/ptic-private-assets"  (default below)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_REPO = "shikharr27/ptic-private-assets"                        # filled in when the Space is published
ROOT = Path(__file__).resolve().parent
NEEDED = ["artifacts/deploy/gate_params.json", "artifacts/deploy/model/CURRENT", "artifacts/torch_cache/hub/checkpoints/resnet18-f37072fd.pth",
          "artifacts/demo_pack/examples.json", "data/metadata/manifest.csv", "reports/11_gradcam_classic.json"]


def missing() -> list[str]:
    gone = [rel for rel in NEEDED if not (ROOT / rel).exists()]
    if not list((ROOT / "artifacts/deploy/model").glob("*/member_*.pt")):
        gone.append("artifacts/deploy/model/*/member_*.pt")
    return gone


def token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    secret = Path("/run/secrets/HF_TOKEN")
    return secret.read_text().strip() if secret.exists() else None


def main() -> int:
    if not missing():
        print("assets already in place")
        return 0
    repo = os.environ.get("PTIC_ASSETS_REPO") or DEFAULT_REPO
    tok = token()
    if repo.startswith("__") or not tok:
        print("cannot fetch the private assets: " + ("the repo id is not set (PTIC_ASSETS_REPO)" if repo.startswith("__") else
              "no token (add the Space secret HF_TOKEN, a read token for the private repo)"))
        return 1
    from huggingface_hub import snapshot_download
    print(f"downloading the private assets from {repo} ...")
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(ROOT), token=tok,
                      ignore_patterns=["README.md", ".gitattributes"])          # keep this Space's own README
    gone = missing()
    if gone:
        print("download finished but these are still missing: " + ", ".join(gone))
        return 1
    print("assets in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
