---
title: PTIC Thyroid Patch Demo
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Research demo - thyroid patch classifier with Grad-CAM
---

# PTIC thyroid patch demo

A research demo of a small convolutional network that reads one H&E-stained thyroid tissue patch and answers
**"papillary thyroid carcinoma detected"**, **"no papillary thyroid carcinoma detected"** or **"uncertain"**, with the model's own
confidence and a Grad-CAM heat map of where in the patch the evidence came from. Before scoring, eleven input checks refuse images
that do not look like the training scans (other scanners, colour casts, blur, wrong magnification, non-tissue).

**Not a medical device. Not for diagnosis.** The page describes how a patch of tissue looks to a model trained on 1,006 hand-picked
patches from 24 patients scanned at one hospital on one scanner. It says nothing about a patient.

* Pick a sample image and press **Analyze the image**, or upload your own patch (JPEG, PNG or TIFF; choose 40x or 20x).
* Images from other hospitals or scanners will very likely be **refused** by the checks. That is intended.
* The free CPU hardware scores an image in several seconds. If the Space has been idle it first needs about a minute to wake up.
* Uploaded images are read into memory, scored and dropped: nothing is written to disk and no image content is logged.

Source code, method and honest limitations: **https://github.com/shikhar2712/ptic-thyroid-classifier**

The model weights and the sample images are kept in a private repository and are not part of this Space's files.

**Credits.** Training data and sample images: PathoTiroid Dataset Indonesian Collection (PTIC), Shabrina et al., Zenodo
10.5281/zenodo.18827019, CC BY-SA 4.0. Typeface: IBM Plex Sans (SIL OFL).
