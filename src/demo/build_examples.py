"""Build the web page's sample set in artifacts/demo_pack/ (examples.json, thumbnails and previews).

Since 2026-10-29 the page shows only images the model answers correctly, and PTC-like ones whose Grad-CAM shows some to strong
evidence. Which images, and why, is decided by src/demo/select_examples.py (rule in its docstring; result in
artifacts/demo_pack/selection.json). This script turns that selection into the pack: the PTC-like images, then the not PTC-like
ones, each from a different patient (now the six images the page showed before 2026-10-29: 3 and 3). PTIC images are referenced in
place (never copied or modified); only previews are new.
The earlier 13-sample build (random PTIC images, the two patient-1 cases, two outside scans, three simulated problems) is in
backups/demo_before_two_images_2026-10-29/code/build_examples.py; the images of those samples are still in the pack folder, unused.

  .venv/bin/python -I src/demo/select_examples.py       # first (CPU, a few minutes)
  .venv/bin/python -I src/demo/build_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "demo_pack"


def save_views(src: Path, ex_id: str) -> None:
    im = Image.open(src).convert("RGB")
    for name, width in (("thumb", 240), ("preview", 960)):
        v = im.resize((width, round(im.height * width / im.width)), Image.BOX)
        v.save(OUT / f"{ex_id}_{name}.jpg", quality=88)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((OUT / "selection.json").read_text())
    man = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "manifest.csv", keep_default_na=False, na_values=[])
    examples = []
    for cls, ids in (("PTC-like", sel["ptc_like"]), ("non-PTC-like", sel["not_ptc_like"])):
        for iid in ids:
            pid = int(man.loc[man.image_id == iid, "patient_id"].iloc[0])
            assert pid != 1, "patient 1 is never a sample"
            examples.append({"id": f"ptic_{len(examples) + 1}", "group": "PTIC image", "title": f"PTIC, patient {pid}",
                             "file": f"data/raw/images/{iid}", "scale": {"magnification": "40x"}, "dataset_label": cls,
                             "note": "From the training dataset. Scored only by the models that never saw this patient."})
    assert len({e["title"] for e in examples}) == len(examples), "one image per patient"
    for ex in examples:
        save_views(PROJECT_ROOT / ex["file"], ex["id"])
    (OUT / "examples.json").write_text(json.dumps({"selection": "artifacts/demo_pack/selection.json", "built_by": "src/demo/build_examples.py",
                                                   "examples": examples}, indent=1))
    for ex in examples:
        print(f"{ex['id']:8s} {ex['dataset_label']:13s} {ex['title']:18s} {ex['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
