"""The words of the demo's result as plain functions, with no web framework: used by the Streamlit front end.

A port of the headline, confidence rule and "About this result" texts of src/demo/static/app.js (the original, which the owner
approved). tests/test_demo_answers.py checks that the key sentences are the same in both, so the two front ends cannot drift apart.
Nothing here ever shows a patient number: `known_ptic_image` is read only for the dataset label.
"""
from __future__ import annotations

import math
from typing import Any

CHECK_NAMES = {
    "scale_known": "Pixel size known", "magnification_sufficient": "Magnification high enough", "size_sufficient": "Image large enough",
    "enough_tissue": "Mostly tissue, not glass", "no_large_dark_areas": "No large dark areas", "colour_present": "Colour image",
    "stain_looks_like_HE": "Colours typical of H&E", "white_balance_neutral": "Neutral white balance", "noise_low": "Low noise",
    "in_focus": "In focus", "appearance_like_training": "Looks like the training scans",
}
PTC, NON_PTC = "PTC", "Non-PTC"                          # short class names (tiles, sentences); the predictor's answers stay "PTC-like" / "not PTC-like"
SHORT = {"PTC-like": PTC, "not PTC-like": NON_PTC}
VERDICT = {"PTC-like": ("Papillary thyroid carcinoma detected", "ptc"),
           "not PTC-like": ("No papillary thyroid carcinoma detected", "neg"),
           "uncertain": ("Uncertain", "unc")}
NOT_A_DIAGNOSIS = ("This tool describes how a patch of tissue looks, not a patient's diagnosis. It is for research use only and is not a "
                   "medical device.")
GRADCAM_KEY = ("Blue: no evidence for PTC. Yellow to red: some to strong evidence. The grey strips at the right and bottom edge are parts "
               "the model never reads.")


def confidence(result: dict[str, Any]) -> str:
    """The model's own score for its answer as a percentage with one decimal, never above 99.9: a network's output is not certainty."""
    p = result["probability_ptc_like"] if result["answer"] == "PTC-like" else 1 - result["probability_ptc_like"]
    return f"{min(99.9, math.floor(p * 1000 + 0.5) / 10):.1f}%"          # floor(x + .5): rounds halves up, like the page's Math.round


def has_confidence(result: dict[str, Any]) -> bool:
    """A percentage only goes with a PTC or Non-PTC answer."""
    return result.get("status") == "ok" and result.get("answer") != "uncertain"


def headline(result: dict[str, Any]) -> tuple[str, str]:
    """(text, kind) of the result's headline; kind is ptc, neg, unc or ref."""
    if result.get("status") != "ok":
        return "Refused", "ref"
    return VERDICT.get(result["answer"], (result["answer"], "unc"))


def dataset_label(example: dict[str, Any]) -> str:
    return PTC if example.get("dataset_label") == "PTC-like" else NON_PTC


def topics(result: dict[str, Any]) -> list[tuple[str, str, list[str | list[str]]]]:
    """The questions that apply to this result, each as (id, question, blocks). A block is a paragraph (str) or a bullet list (list)."""
    ok = result.get("status") == "ok"
    out: list[tuple[str, str, list[str | list[str]]]] = []
    if ok:
        if result["answer"] == "PTC-like":
            meaning: list[str | list[str]] = [f'"Papillary thyroid carcinoma detected" ({PTC}) means the tissue in this image looks like papillary thyroid '
                                              "carcinoma to the model. Papillary thyroid carcinoma is the most common type of thyroid cancer.", NOT_A_DIAGNOSIS]
        elif result["answer"] == "not PTC-like":
            meaning = [f'"No papillary thyroid carcinoma detected" ({NON_PTC}) means the tissue in this image does not look like papillary thyroid '
                       "carcinoma to the model.", NOT_A_DIAGNOSIS]
        else:
            meaning = ["The model is not clear enough about this image, so it gives neither of its two answers: papillary thyroid carcinoma "
                       "detected, or no papillary thyroid carcinoma detected.", NOT_A_DIAGNOSIS]
        out.append(("meaning", "What does this result mean?", meaning))
    else:
        reasons = list(result.get("refusal_reasons") or []) or ["the image could not be read"]
        out.append(("refused", "Why was it refused?", ["It did not pass the quality checks:", reasons,
                                                         "The model only answers for images that look like its training scans; it refuses the others instead of guessing."]))
    if ok and result.get("ensemble"):
        if result["answer"] == "uncertain":
            sure: list[str | list[str]] = ['The model is not sure about this image: it cannot choose between "papillary thyroid carcinoma detected" '
                                           'and "no papillary thyroid carcinoma detected".']
        else:
            sure = [f"The model's confidence in this answer is {confidence(result)}.",
                    "That is how strongly the model favours its own answer. It is not the chance that a patient has cancer, and a model can be "
                    "confident and still wrong."]
        out.append(("sure", "How sure is the model?", sure))
    if ok and (result.get("heatmap") or {}).get("made"):
        out.append(("where", "Where did the model look?",
                    [f"The Grad-CAM image colours each part of the picture by how much it pushed the model towards papillary thyroid carcinoma ({PTC}): "
                     "blue means none, yellow to red means strong.",
                     "It shows where, not why. It cannot rule out that the model reacts to the colour of its training hospital's stain or scanner "
                     "instead of the cells."]))
    known = result.get("known_ptic_image")
    if known:
        lab = "PTC-like" if known.get("dataset_label") == "PTC-like" else "not PTC-like"          # the predictor's names, to compare with the answer
        block: list[str | list[str]] = [f"This image comes from the PTIC dataset, where pathologists labelled it {SHORT[lab]}."]
        if not ok:
            block.append("It was refused, so there is no answer to compare.")
        else:
            if result["answer"] == "uncertain":
                block.append("The model is uncertain about it, so there is nothing to compare.")
            elif result["answer"] == lab:
                block.append("The model agrees.")
            else:
                block.append(f"The model disagrees: it answers {SHORT.get(result['answer'], result['answer'])}.")
            block.append("The models that scored it never saw this image's patient, so this is a fair test.")
        out.append(("label", "Does it match the dataset label?", block))
    checks = result.get("checks") or {}
    if checks:
        bad = [(n, c) for n, c in checks.items() if not c.get("passed")]
        if not bad:
            body: list[str | list[str]] = [f"Yes, all {len(checks)} checks passed:", [CHECK_NAMES.get(n, n) for n in checks]]
        else:
            body = [f"No, {len(bad)} of {len(checks)} checks failed:",
                    [CHECK_NAMES.get(n, n) + (f" ({c['reason']})" if c.get("reason") else "") for n, c in bad]]
        out.append(("checks", "Did the image pass the quality checks?", body))
    return out
