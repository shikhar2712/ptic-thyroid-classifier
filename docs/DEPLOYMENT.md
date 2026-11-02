# Deployment notes

The same Python code (input gate, 15-model ensemble, Grad-CAM) has two front ends:

| Front end | Where | Used for |
|---|---|---|
| FastAPI service with a hand-written page (`src/demo`) | local, or as a Docker image (`deploy/space/`) | the internal demo and the JSON API |
| Streamlit host (`deploy/streamlit/streamlit_app.py`) | Streamlit Community Cloud | the public demo: the same page, unchanged |

The Streamlit host is the public one because it is free to host from a GitHub repository. Hugging Face now charges for Docker Spaces on CPU
(HTTP 402, "requires a PRO subscription" in November 2026), and static Spaces cannot run a model. The Docker files stay in `deploy/space/`
for anyone with a PRO account; they were verified locally but never built on Hugging Face.

## What is where

| Place | Visibility | Holds |
|---|---|---|
| This GitHub repository | public | code, tests, configuration, this documentation. No data, weights, patient tables or tissue images |
| Streamlit Community Cloud app | public | runs the code of this repository |
| Hugging Face model repository | **private** | the 15 ensemble members, the input-check statistics, ImageNet ResNet-18 weights, six sample images |

The app downloads the private repository at its first start, with a read-only token stored as a **Streamlit secret** (never in a file of
this repository), into the app folder, or into a temporary folder when that is not writable (`src/demo/assets.py`). The download and
the model load are cached for the life of the process; one analysis runs at a time, which also bounds memory.

## How the same page runs inside Streamlit

Streamlit has no web server to hand the page its `api/...` requests, so the page is not rewritten: it runs unchanged (`src/demo/static`)
inside a Streamlit *component*. `deploy/streamlit/ui_shim.js` is loaded in front of the page's own script and answers its `fetch("api/...")`
calls and `<img src="api/...">` images by sending them, in batches, through the component channel to Python, where `src/demo/rpc.py`
answers them with the same paths and JSON as the FastAPI service (`tests/test_demo_rpc.py`). The page, its styles, its fonts and its
wording are the files of the FastAPI demo; only the transport differs. Streamlit's own header and padding are hidden so that the page
fills the window.

## Deploying

1. On huggingface.co create a **fine-grained token** with read access to the private repository only.
2. On share.streamlit.io choose *New app*, pick this repository, branch `main`, main file `deploy/streamlit/streamlit_app.py`.
3. In *Advanced settings* choose **Python 3.12** (the two PyTorch lines in `deploy/streamlit/requirements.txt` are CPU wheels for 3.12) and
   put the secrets in the box:

   ```toml
   HF_TOKEN = "hf_..."
   PTIC_ASSETS_REPO = "<user>/<private repository>"   # optional if it is the default in streamlit_app.py
   ```

`deploy/streamlit/requirements.txt` sits next to the entry point, so Community Cloud uses it instead of the root `requirements.txt`. It
pins the CPU build of PyTorch by URL, so no installer can pick the multi-gigabyte CUDA build from PyPI. `.streamlit/config.toml` turns the
file watcher off (it trips over `torch.classes`), caps uploads at 25 MB and hides error tracebacks from visitors.

## Verified before publishing

On the machine that built the project, in a brand-new Python 3.12 environment made with `uv` (the installer Community Cloud tries first)
from `deploy/streamlit/requirements.txt` alone, run on 2 CPU cores, with the model downloaded from the private repository:

* the install resolves to `torch 2.11.0+cpu` and installs no `nvidia-*`, `cuda` or `triton` package (1.3 GB on disk);
* a browser (Playwright) drove the page inside Streamlit: two sample images give "Papillary thyroid carcinoma detected" and "No papillary
  thyroid carcinoma detected" at 99.9% confidence with their Grad-CAM and the five "About this result" questions; a non-tissue upload
  is refused with reasons; an upload that is not a known sample is scored by all 15 models; the dark theme toggle works; no patient
  number appears anywhere on the page;
* time from click to answer, measured in the browser on 2 cores: about 10 seconds for a sample (3 models) and about 29 seconds for an
  upload (15 models), with the heat map;
* peak memory of the process (`VmHWM`) after all of the above: **1.25 GB**. Streamlit's own documentation lists up to about 2.7 GB per
  app for Community Cloud (a February 2024 figure that may have changed); the limit in force was not checked;
* the six built-in samples give the same answers as the GPU run the project was developed on, with probabilities within 0.0001 on the
  released CPU build of PyTorch;
* the test suite passes on a bare checkout (91 passed, 10 skipped: they need the dataset, trained artifacts or Streamlit), and a test checks that the app says what is missing instead of crashing when the model files
  or the secret are absent.

Not verified until the app is deployed: how the page sizes itself inside Community Cloud's own frame, Community Cloud's actual memory, CPU and disk limits, the first start's download time there, and
how long the app takes to wake after it has been put to sleep for inactivity.

## Privacy

* The Streamlit host never receives a patient number or image id: its list of known sample files holds only a file hash, a
  cross-validation fold and the dataset's class.
* The FastAPI service has a public mode (`PTIC_DEMO_PUBLIC=1`) that removes patient numbers, image ids and example titles from every
  response, and `PTIC_DEMO_FRAME_ANCESTORS` to allow embedding by a named host; `deploy/smoke_test.py <url>` checks a running FastAPI
  service for both.
* The reference heat-map figures of the internal demo have patient numbers drawn into them, so they are not part of any public
  deployment.
* Uploaded images are read into memory, scored and dropped.
