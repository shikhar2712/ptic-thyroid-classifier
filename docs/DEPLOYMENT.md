# Deployment notes

The demo is one FastAPI service (`src/demo/app.py`) with a hand-written front end (`src/demo/static`). Three places hold the project:

| Where | Visibility | Holds |
|---|---|---|
| GitHub repository | public | code, tests, configs, this documentation. No data, weights, patient tables or tissue images |
| Hugging Face Space (Docker SDK, free CPU) | public | the same code, a Dockerfile, a model card. **No weights** |
| Hugging Face model repository | **private** | the 15 ensemble members, the input-check statistics, ImageNet ResNet-18 weights, six sample images |

The Space reads the private repository with a read-only token that is stored as a **Space secret** (never in a file or in the image's
public layers), at build time when the build is given the secret and otherwise when the container starts (`deploy/space/fetch_assets.py`).

## Why a Docker Space and not Streamlit

The service is already an HTTP API plus a purpose-built page. A Docker Space runs it unchanged on port 7860. Streamlit would mean
rewriting the interface, and Streamlit Community Cloud's memory (about 1 GB) is too small for PyTorch with a 15-model ensemble; the free
Space gives 2 vCPU and 16 GB.

## Public mode

Environment variables read by the app (all optional; the defaults are the internal setup):

| Variable | Public Space | Effect |
|---|---|---|
| `PTIC_DEMO_PUBLIC` | `1` | no patient number, image id or example title in any response |
| `PTIC_DEMO_FRAME_ANCESTORS` | `https://huggingface.co https://*.hf.space` | lets the Space page embed the app; the default is `'none'` |
| `PTIC_DEMO_DEVICE` | `cpu` | |
| `PTIC_DEMO_MAX_MB` / `PTIC_DEMO_MAX_MPIX` | `25` / `40` | upload limits; whole slides are refused |
| `OMP_NUM_THREADS` | `2` | fixed before PyTorch starts, as the free hardware has 2 vCPU |

Uploads are read into memory, scored and dropped. One prediction runs at a time. The internal demo's reference heat-map figures have patient
numbers drawn into them, so the public Space does not get them: the page hides its "Reference heat maps" panel when the server has none.

## Verified before publishing

The exported Space tree was run in a clean virtual environment (no system packages) with the released CPU build of PyTorch
(2.11.0+cpu), pinned to 2 cores, against the GPU run the project was developed on:

* the six built-in samples give the same answers; probabilities differ by at most 0.0001 (rounding in the fourth decimal);
* a re-encoded upload scored by all 15 members gives the same answer;
* `deploy/smoke_test.py` passes 22 of 22 checks (page and headers, health, sample list, each sample with Grad-CAM, a non-tissue upload
  refused, a non-image refused, no patient field in any response);
* the test suite passes on a bare checkout: 85 passed, 9 skipped (they need the dataset or the trained artifacts), 0 failed.

Latency on 2 cores, measured: about 6 seconds for a built-in sample with its heat map (3 models), about 25 seconds for an upload (15 models).

## Checking a running deployment

```bash
python3 deploy/smoke_test.py https://<user>-<space>.hf.space           # about a minute
python3 deploy/smoke_test.py https://<user>-<space>.hf.space --quick   # page, headers, health, sample list only
```

## Updating

* **Code:** push to the Space repository; the Space rebuilds.
* **Weights:** upload to the private repository, then *Factory reboot* the Space so the build fetches them again.
* **Free-tier behaviour:** a Space that has been idle for a while is put to sleep and wakes on the next visit (about a minute).
