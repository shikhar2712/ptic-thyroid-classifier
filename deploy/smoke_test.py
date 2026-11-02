#!/usr/bin/env python3
"""End-to-end check of a running PTIC demo, from the outside, with the standard library only.

  python3 deploy/smoke_test.py https://<user>-<space>.hf.space
  python3 deploy/smoke_test.py http://127.0.0.1:8798 --quick

Checks: the page and its security headers, /api/health, the sample list, then (unless --quick) each of the six samples through the
public API with Grad-CAM, and an upload that must be refused. For a public deployment it also checks that no patient number, image id
or example title is in any response. Exit status 0 only when every check passed. Costs about a minute on free CPU hardware.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

EXPECTED = {f"ptic_{i}": "PTC-like" if i <= 3 else "not PTC-like" for i in range(1, 7)}   # what the six built-in samples must answer
FORBIDDEN_KEYS = {"patient_id", "image_id", "title"}
PATIENT_NUMBER = re.compile(r"patient\s*#?\s*\d", re.I)
results: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> None:
    results.append((bool(ok), what))
    print(("PASS  " if ok else "FAIL  ") + what, flush=True)


def get(url: str, timeout: float = 90):
    return urllib.request.urlopen(urllib.request.Request(url), timeout=timeout)


def post(url: str, data: bytes = b"", timeout: float = 300) -> dict:
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/octet-stream"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def walk_keys(o) -> set[str]:
    if isinstance(o, dict):
        return set(o) | set().union(*(walk_keys(v) for v in o.values())) if o else set()
    if isinstance(o, list):
        return set().union(*(walk_keys(v) for v in o)) if o else set()
    return set()


def gradient_png(w: int = 512, h: int = 512) -> bytes:
    """A smooth grey-blue gradient: not tissue, so the input checks must refuse it."""
    raw = b"".join(b"\x00" + bytes(v for x in range(w) for v in (x * 255 // w, y * 255 // h, 180)) for y in range(h))
    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="base URL of the demo, without a trailing slash")
    ap.add_argument("--quick", action="store_true", help="skip the six scored samples and the refused upload")
    ap.add_argument("--allow-patient-fields", action="store_true", help="internal (non-public) server: do not require the redactions")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    r = get(base + "/")
    page = r.read().decode("utf-8", "replace")
    csp = r.headers.get("content-security-policy", "")
    check(r.status == 200 and "Not a medical device" in page, "page loads and says it is not a medical device")
    check("default-src 'self'" in csp and "object-src 'none'" in csp, "content security policy is in place")
    if not a.allow_patient_fields:
        check("frame-ancestors" in csp and "huggingface.co" in csp, "page may be framed by huggingface.co (Space page)")
    h = json.load(get(base + "/api/health"))
    check(h.get("status") == "ok" and h.get("heatmaps") is True, f"health ok, model {h.get('model')}, device {h.get('device')}, heat maps on")
    want_figs = 3 if a.allow_patient_fields else 0                  # the public server must not serve figures that have patient numbers in them
    check(h.get("examples") == 6 and h.get("figures") == want_figs and h.get("known_ptic_images", 0) >= 6,
          f"6 samples, {want_figs} reference figures and the sample fingerprints are loaded")
    ex = json.load(get(base + "/api/examples"))["examples"]
    check([e["id"] for e in ex] == list(EXPECTED), "sample list is ptic_1 .. ptic_6")
    for e in ex:
        img = get(f"{base}/api/examples/{e['id']}/thumb").read()
        if img[:2] != b"\xff\xd8":
            check(False, f"thumbnail of {e['id']} is a JPEG")
            break
    else:
        check(True, "all six thumbnails are JPEGs")
    if not a.allow_patient_fields:
        check(not (FORBIDDEN_KEYS & walk_keys(ex)), "sample list carries no patient number, image id or title")
    if a.quick:
        bad = [w for ok, w in results if not ok]
        print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
        return 1 if bad else 0

    for ex_id, want in EXPECTED.items():
        t = time.time()
        r = post(f"{base}/api/examples/{ex_id}/predict?explain=1")
        dt = time.time() - t
        hm = r.get("heatmap") or {}
        ok = r["status"] == "ok" and r["answer"] == want and hm.get("made") is True and r.get("ensemble", {}).get("members") == 3
        check(ok, f"{ex_id}: {r['answer']} (P={r['probability_ptc_like']}), 3 fair-scoring members, Grad-CAM made, {dt:.1f}s")
        if not a.allow_patient_fields:
            text = json.dumps({k: v for k, v in r.items() if k != "heatmap"})                  # the heat map is base64 image data
            check(not (FORBIDDEN_KEYS & walk_keys(r)) and not PATIENT_NUMBER.search(text),
                  f"{ex_id}: response carries no patient number or image id")
    r = post(f"{base}/api/predict?magnification=40x", gradient_png())
    check(r["status"] == "refused" and r["refusal_reasons"], f"a non-tissue upload is refused: {r.get('refusal_reasons', [''])[0][:70]}")
    try:
        r = post(f"{base}/api/predict?magnification=40x", b"not an image")
        check(r["status"] == "refused" and any("not a readable image" in x for x in r["refusal_reasons"]), "a file that is not an image is refused as unreadable")
    except urllib.error.HTTPError as exc:
        check(exc.code in (400, 415, 422), f"a file that is not an image is rejected with HTTP {exc.code}")

    bad = [w for ok, w in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + ("" if not bad else "; FAILED: " + "; ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
