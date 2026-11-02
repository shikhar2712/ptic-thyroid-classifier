"""The wording of the result as plain Python (used by the Streamlit front end) and its agreement with the page's JavaScript."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.demo import answers as A  # noqa: E402

JS = (Path(__file__).resolve().parents[1] / "src" / "demo" / "static" / "app.js").read_text()


def res(answer="PTC-like", p=0.9995, **kw):
    return {"status": "ok", "answer": answer, "probability_ptc_like": p, "ensemble": {"members": 3}, "checks": {}, **kw}


def test_confidence_is_the_models_own_score_for_its_answer_and_never_above_99_9():
    assert A.confidence(res("PTC-like", 0.9995)) == "99.9%" and A.confidence(res("PTC-like", 1.0)) == "99.9%"
    assert A.confidence(res("not PTC-like", 0.0043)) == "99.6%" and A.confidence(res("not PTC-like", 0.0)) == "99.9%"
    assert A.confidence(res("PTC-like", 0.8123)) == "81.2%" and A.confidence(res("not PTC-like", 0.15)) == "85.0%"
    assert A.has_confidence(res("PTC-like")) and not A.has_confidence(res("uncertain", 0.5))
    assert not A.has_confidence({"status": "refused", "answer": None})


def test_headlines_use_the_full_names_with_detected_as_suffix():
    assert A.headline(res("PTC-like")) == ("Papillary thyroid carcinoma detected", "ptc")
    assert A.headline(res("not PTC-like", 0.0)) == ("No papillary thyroid carcinoma detected", "neg")
    assert A.headline(res("uncertain", 0.5)) == ("Uncertain", "unc")
    assert A.headline({"status": "refused"}) == ("Refused", "ref")


def test_topics_for_an_answered_sample_that_matches_its_label():
    r = res("PTC-like", heatmap={"made": True}, known_ptic_image={"dataset_label": "PTC-like", "test_fold": 0},
            checks={"in_focus": {"passed": True}, "noise_low": {"passed": True}})
    t = {i: (q, b) for i, q, b in A.topics(r)}
    assert list(t) == ["meaning", "sure", "where", "label", "checks"]
    assert "The model agrees." in t["label"][1] and "99.9%" in t["sure"][1][0]
    assert t["checks"][1][0] == "Yes, all 2 checks passed:" and t["checks"][1][1] == ["In focus", "Low noise"]
    assert "patient_id" not in str(A.topics(r))


def test_topics_for_a_disagreement_a_failed_check_and_a_refusal():
    r = res("not PTC-like", 0.01, known_ptic_image={"dataset_label": "PTC-like", "test_fold": 1},
            checks={"in_focus": {"passed": False, "reason": "blurred"}})
    t = {i: b for i, _, b in A.topics(r)}
    assert "The model disagrees: it answers Non-PTC." in t["label"] and t["checks"][1] == ["In focus (blurred)"]
    refused = {"status": "refused", "answer": None, "refusal_reasons": ["colour cast"], "checks": {}, "known_ptic_image": {"dataset_label": "PTC-like"}}
    t = {i: b for i, _, b in A.topics(refused)}
    assert list(t) == ["refused", "label"] and t["refused"][1] == ["colour cast"] and "It was refused, so there is no answer to compare." in t["label"]
    assert "The models that scored it never saw this image's patient, so this is a fair test." not in t["label"]
    assert A.topics({"status": "refused", "refusal_reasons": []})[0][2][1] == ["the image could not be read"]


@pytest.mark.parametrize("fragment", [
    A.NOT_A_DIAGNOSIS, "Papillary thyroid carcinoma detected", "No papillary thyroid carcinoma detected",
    "means the tissue in this image looks like papillary thyroid carcinoma to the model. Papillary thyroid carcinoma is the most common type of thyroid cancer.",
    "does not look like papillary thyroid carcinoma to the model.",
    "The model is not clear enough about this image, so it gives neither of its two answers",
    "That is how strongly the model favours its own answer. It is not the chance that a patient has cancer, and a model can be confident and still wrong.",
    "It shows where, not why. It cannot rule out that the model reacts to the colour of its training hospital's stain or scanner instead of the cells.",
    "The model only answers for images that look like its training scans; it refuses the others instead of guessing.",
    "The models that scored it never saw this image's patient, so this is a fair test.",
    "Mostly tissue, not glass", "Looks like the training scans", "Colours typical of H&E",
])
def test_key_sentences_are_the_same_as_in_the_pages_javascript(fragment):
    assert fragment in JS
