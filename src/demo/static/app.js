"use strict";
// Thyroid patch check: the page. It talks only to this server (relative URLs, so it also works behind a path prefix).
// Flow: an image goes on the stage (a sample image from the strip, or an upload); "Analyze the image" sends it, an animation runs
// while the server works, then the answer, the image's heat map and questions about the result appear.
const $ = (id) => document.getElementById(id);
const NAMES = {
  scale_known: "Pixel size known", magnification_sufficient: "Magnification high enough", size_sufficient: "Image large enough",
  enough_tissue: "Mostly tissue, not glass", no_large_dark_areas: "No large dark areas", colour_present: "Colour image",
  stain_looks_like_HE: "Colours typical of H&E", white_balance_neutral: "Neutral white balance", noise_low: "Low noise",
  in_focus: "In focus", appearance_like_training: "Looks like the training scans",
};
const PTC = "PTC", NON_PTC = "Non-PTC";   // the short class names (tiles, sentences); the server's answers stay "PTC-like" / "not PTC-like"
const SHORT = { "PTC-like": PTC, "not PTC-like": NON_PTC };
// The headline of the result: the full names, with "detected" as suffix.
const VERDICT = { "PTC-like": ["Papillary thyroid carcinoma detected", "d-ptc"], "not PTC-like": ["No papillary thyroid carcinoma detected", "d-neg"],
                  "uncertain": ["Uncertain", "d-unc"] };

// Shows the headline. A long answer is set smaller (and wraps on a narrow card); Uncertain, Refused and Not analyzed keep the big size.
function setVerdict(text, cls) {
  $("verdict-text").textContent = text; $("verdict-dot").className = "dot " + cls;
  $("verdict").classList.toggle("long", text.length > 14);
}
const GROUP_ORDER = ["PTIC image", "Known hard case", "Outside scan", "Simulated problem"];   // the order of the sample images
const GROUP_TAG = { "PTIC image": "PTIC", "Known hard case": "Hard case", "Outside scan": "Other hospital",
                    "Simulated problem": "Simulated problem" };   // the short tag under each tile's name
const CODE_KEY = "ptic-demo-access-code";   // where this browser remembers the access code (localStorage)
const MIN_MS = 2000;                        // the analysis animation runs at least this long (0.4 s under reduced motion)
const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
const oneColumn = window.matchMedia("(max-width: 1055px)");
let health = null;
let limitMb = 40;
let busy = false;
let mode = "empty";                         // the stage: empty, ready (an image waits for Analyze), busy or done
let current = null;                         // the image on the stage: { kind: "sample" | "upload", src, name, meta, ex | file }
let answer = null;                          // the server's answer for the image on the stage
let objectUrl = null;                       // the local preview of an upload
let blobUrls = [];                          // the heat-map pictures of the current answer
let seq = 0;                                // bumped whenever the stage changes, so a late answer for an earlier image is dropped

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

function scaleParam() {
  const sel = document.querySelector("input[name=scale]:checked");
  if (!sel) return null;
  if (sel.value !== "mpp") return "magnification=" + encodeURIComponent(sel.value);
  const v = parseFloat($("mpp").value);
  return v > 0 ? "mpp=" + v : null;
}

function authHeaders() {
  const token = $("token").value.trim();
  return health && health.token_required && token ? { Authorization: "Bearer " + token } : {};
}

// The access code can come from a link ending in #code=... (browsers never send the part after # to the server).
// This browser remembers it once the server has accepted it and forgets it when the server rejects it.
function restoreCode() {
  let code = "";
  try {
    const m = /(?:^#|&)code=([^&]*)/.exec(location.hash);
    if (m) {
      code = decodeURIComponent(m[1]);
      history.replaceState(null, "", location.pathname + location.search);   // out of sight, e.g. when sharing the screen
    }
    if (!code) code = localStorage.getItem(CODE_KEY) || "";
  } catch (e) { /* storage blocked or a malformed link: type the code instead */ }
  $("token").value = code;
}

function rememberCode(code) {
  try {
    if (code) localStorage.setItem(CODE_KEY, code); else localStorage.removeItem(CODE_KEY);
  } catch (e) { /* storage blocked: the code is not remembered */ }
}

async function loadHealth() {
  try {
    const r = await fetch("api/health", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    health = await r.json();
    $("about-model").textContent = health.model;
    limitMb = Math.round(health.max_upload_mb);
    $("limit").textContent = String(limitMb);
    $("token-row").hidden = !health.token_required;
    $("refs").hidden = !health.figures;
  } catch (e) {
    health = null;
  }
  syncFoot();
}

// --- the sample images: numbered in the strip's order ----------------------------------------------------------------------------

// The class the dataset's pathologists gave an image (empty for outside scans and simulated problems, which have none).
function datasetLabel(ex) {
  return !ex.dataset_label ? "" : ex.dataset_label === "PTC-like" ? PTC : NON_PTC;
}

// One tile of the moving strip. The strip shows every sample `copies` times so that it can loop without a jump on any screen width;
// the extra copies are for the eye and the mouse only (hidden from screen readers and from the Tab key).
function tile(s, twin) {
  const b = el("button", undefined, "ex");
  b.type = "button";
  b.dataset.id = s.ex.id;
  b.setAttribute("aria-pressed", "false");
  if (twin) { b.setAttribute("aria-hidden", "true"); b.tabIndex = -1; }
  const img = el("img");
  img.src = "api/examples/" + encodeURIComponent(s.ex.id) + "/thumb"; img.alt = ""; img.draggable = false;
  const sub = el("span", undefined, "s");
  sub.append(el("span", GROUP_TAG[s.ex.group] || s.ex.group));
  if (datasetLabel(s.ex)) sub.append(el("b", "Label: " + datasetLabel(s.ex)));
  b.append(img, el("span", "Sample image " + s.n, "t"), sub);
  b.addEventListener("click", () => loadSample(s));
  return b;
}

async function loadExamples() {
  try {
    const r = await fetch("api/examples", { cache: "no-store" });
    if (!r.ok) return;
    const list = (await r.json()).examples || [];
    list.sort((a, b) => GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group));
    const samples = list.map((ex, i) => ({ ex, n: i + 1 }));
    const copyW = samples.length * 200;                            // one copy of the tiles and their gaps, at most
    copies = Math.max(COPIES_MIN, Math.ceil(Math.max(screen.width, innerWidth) / copyW) + 1);   // the loop must outlast the widest screen
    for (let copy = 0; copy < copies; copy++) for (const s of samples) $("examples").append(tile(s, copy > 0));
    $("examples-card").hidden = samples.length === 0;
  } catch (e) { /* the sample images are optional */ }
}

// The strip runs across the whole width and moves slowly to the left on its own. It stops while the mouse is on it, while a tile
// has keyboard focus and for a few seconds after a touch or a wheel turn; under prefers-reduced-motion it never moves on its own.
// The mouse wheel glides it either way. It loops: the position jumps by one copy of the tiles, which looks the same, whenever it
// leaves the first copy; with enough copies (at least three) the strip never runs out, even on a wide screen with few samples. The position is kept as a decimal number, because
// browsers round scrollLeft to whole pixels.
const SPEED = 24;                                  // pixels per second
const COPIES_MIN = 3;
let copies = COPIES_MIN;                           // set when the samples arrive
const strip = { hover: false, focus: false, hold: 0, pos: 0, to: null, last: 0 };

function stripStep(t) {
  const box = $("strip"), tiles = $("examples").children, n = tiles.length / copies;
  const dt = strip.last ? Math.min(64, t - strip.last) : 0;
  strip.last = t;
  const loop = n >= 1 ? tiles[n].offsetLeft - tiles[0].offsetLeft : 0;   // one copy of the tiles and its gap
  let moved = true;
  if (strip.to !== null) {                         // a wheel turn: glide there (checked first: the wheel turns on hover)
    strip.pos += (strip.to - strip.pos) * (calm.matches ? 1 : Math.min(1, dt / 90));
    if (Math.abs(strip.to - strip.pos) < 0.5) { strip.pos = strip.to; strip.to = null; }
  } else if (!strip.hover && !strip.focus && t > strip.hold && !calm.matches && !$("examples-card").hidden) {
    strip.pos += SPEED * dt / 1000;
  } else {
    strip.pos = box.scrollLeft;                    // follow a swipe and keyboard focus
    moved = false;
  }
  if (loop > 8 && (strip.pos >= loop || strip.pos < 0)) {
    const by = strip.pos < 0 ? loop : -loop;
    strip.pos += by;
    if (strip.to !== null) strip.to += by;
    moved = true;
  }
  if (moved) box.scrollLeft = strip.pos;
  requestAnimationFrame(stripStep);
}

function initStrip() {
  const box = $("strip");
  const hold = (ms) => { strip.hold = performance.now() + ms; };
  const glide = (px) => { strip.to = (strip.to ?? strip.pos) + px; };
  // Only a mouse stops it by pointing: on a phone, a tap counts as pointing until the next tap, so the strip would stay still.
  box.addEventListener("pointerenter", (e) => { if (e.pointerType === "mouse") strip.hover = true; });
  box.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") strip.hover = false; });
  // A click also focuses its tile; only keyboard focus (the visible ring) keeps the strip still after the pointer has left.
  box.addEventListener("focusin", (e) => { strip.focus = e.target.matches(":focus-visible"); });
  box.addEventListener("focusout", () => { strip.focus = false; });
  for (const t of ["touchstart", "pointerdown"]) box.addEventListener(t, () => hold(4000), { passive: true });
  // The mouse wheel moves the strip sideways instead of scrolling the page; a sideways touchpad swipe scrolls it natively, and
  // Ctrl + wheel still zooms.
  box.addEventListener("wheel", (e) => {
    hold(4000);
    if (e.ctrlKey || Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
    e.preventDefault();
    glide(e.deltaY * (e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? box.clientWidth : 1));
  }, { passive: false });
  requestAnimationFrame(stripStep);
}

// The reference heat-map pictures (about 3 MB) are downloaded only when their section in About is opened.
function loadFigures() {
  if (!$("refs").open) return;
  for (const img of $("refs").querySelectorAll("img[data-src]")) {
    img.src = img.dataset.src;
    img.removeAttribute("data-src");
  }
}

// --- the stage: the image, then the analysis, then the heat map over it ------------------------------------------------------------

function loadSample(s) {
  const m = s.ex.scale || {};
  const loaded = setImage({ kind: "sample", ex: s.ex, name: "Sample image " + s.n,
                            src: "api/examples/" + encodeURIComponent(s.ex.id) + "/preview",
                            meta: m.magnification ? "scanned at " + m.magnification.replace("x", "×") : m.mpp ? m.mpp + " µm per pixel" : "" });
  if (loaded) bringStage();
}

function loadUpload(file) {
  if (busy || !file) return;
  setImage({ kind: "upload", file, name: file.name, src: URL.createObjectURL(file), meta: (file.size / 1048576).toFixed(1) + " MB" });
}

// On one column the strip sits below everything: bring the stage into view when a sample is picked there.
function bringStage() {
  const r = $("stage-card").getBoundingClientRect();
  if (r.top < 0 || r.top > window.innerHeight / 2) $("stage-card").scrollIntoView({ behavior: calm.matches ? "auto" : "smooth", block: "start" });
}

function freeAnswer() {
  for (const u of blobUrls) URL.revokeObjectURL(u);
  blobUrls = [];
}

function blobUrl(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
  blobUrls.push(url);
  return url;
}

// Put an image on the stage, ready to be analyzed. Nothing is sent to the server until "Analyze the image" is pressed.
function setImage(img) {
  if (busy) return false;
  seq++;
  freeAnswer();
  answer = null;
  if (objectUrl && (!img || img.src !== objectUrl)) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  if (img && img.kind === "upload") objectUrl = img.src;
  current = img;
  hideMap();
  const scan = $("hm-scan");
  scan.onload = scan.onerror = null;
  scan.hidden = true;
  $("drop").hidden = !!img;
  $("stage-note").hidden = true;
  $("stage-title").textContent = img ? img.name : "Image";
  $("stage-meta").textContent = img ? img.meta : "";
  if (img) {
    scan.alt = img.name;
    scan.onload = () => {
      scan.hidden = false;
      $("stage").style.setProperty("--ar", (scan.naturalWidth / scan.naturalHeight).toFixed(4));
      if (img.kind === "upload") $("stage-meta").textContent = img.meta + " · " + scan.naturalWidth + " × " + scan.naturalHeight + " px";
    };
    scan.onerror = () => { $("stage-note").hidden = false; };   // e.g. TIFF: the server reads it, this browser cannot show it
    scan.src = img.src;
  } else {
    scan.removeAttribute("src");
    $("stage").style.removeProperty("--ar");
  }
  for (const b of document.querySelectorAll(".ex")) {
    b.setAttribute("aria-pressed", img && img.kind === "sample" && b.dataset.id === img.ex.id ? "true" : "false");
  }
  mode = img ? "ready" : "empty";
  showEmpty();
  syncFoot();
  return true;
}

function hideMap() {
  $("pair").hidden = true;
  $("pair-scan").removeAttribute("src"); $("pair-heat").removeAttribute("src");
  $("stage").hidden = false;
  document.querySelector(".layout").classList.remove("has-pair");
}

// What sits under the stage follows the mode: the scale (uploads only), the button, the progress steps or the heat map's controls.
function syncFoot() {
  const up = !!current && current.kind === "upload";
  const sel = document.querySelector("input[name=scale]:checked");
  const ready = mode === "empty" || mode === "ready";
  $("scale").hidden = !up || mode === "busy";
  $("mpp-row").hidden = $("scale").hidden || !(sel && sel.value === "mpp");
  $("analyze").hidden = !ready;
  $("go-hint").hidden = !ready;
  $("go-row").hidden = $("scale").hidden && !ready;
  $("analyze").disabled = busy || !health || !current || (up && !scaleParam());
  $("go-hint").textContent = !health ? "The server is not reachable." : !current ? "Upload an image or pick a sample image below."
    : up && !sel ? "Choose how it was scanned: a wrong magnification gives wrong answers."
    : up && !scaleParam() ? "Enter the size of one pixel in micrometres." : "";
  const made = mode === "done" && !!answer && !!(answer.heatmap && answer.heatmap.made);
  $("steps").hidden = !(mode === "busy" || (mode === "done" && !made));   // with no heat map, the steps stay as the summary
  $("viewer-foot").hidden = !made;
  $("no-map").hidden = !(mode === "done" && answer && answer.status === "ok" && !made);
  $("upload-btn").hidden = !current || busy;
}

// A new scale makes the answer for an upload out of date: back to "ready".
function scaleChanged() {
  if (mode === "done" && current && current.kind === "upload") setImage(current);
  else syncFoot();
}

// --- the analysis --------------------------------------------------------------------------------------------------------------

// The three steps under the image. While the server works, the step in progress turns; a tick or a cross appears only when the
// answer has arrived and says so, so a refused image never shows "Quality checks" as passed.
function runSteps() {
  const items = [...$("steps").children];
  const set = (i, state, sub) => { items[i].className = state; items[i].querySelector(".st-sub").textContent = sub || ""; };
  items.forEach((_, i) => set(i, "wait"));
  set(0, "run");
  const timers = [setTimeout(() => { set(0, "seen"); set(1, "run"); }, 650), setTimeout(() => { set(1, "seen"); set(2, "run"); }, 1300)];
  return {
    async finish(res) {
      timers.forEach(clearTimeout);
      if (!res) { items.forEach((_, i) => set(i, "skip")); return; }
      const checks = Object.values(res.checks || {}), passed = checks.filter((c) => c.passed).length;
      const ok = res.status === "ok", made = ok && !!(res.heatmap && res.heatmap.made);
      const final = [
        ok ? ["ok", passed + " of " + checks.length + " passed"] : ["fail", passed < checks.length ? (checks.length - passed) + " failed" : "Refused"],
        ok ? ["ok", "Done"] : ["skip", "Skipped"],
        made ? ["ok", "Ready"] : ["skip", ok ? "Not made" : "Skipped"],
      ];
      for (let i = 0; i < final.length; i++) {
        set(i, final[i][0], final[i][1]);
        await sleep(calm.matches ? 0 : 160);
      }
      await sleep(calm.matches ? 0 : 260);
    },
  };
}

async function analyze() {
  if (busy || !current || $("analyze").disabled) return;
  const img = current, id = ++seq;
  let url, options;
  if (img.kind === "upload") {
    if (img.file.size > limitMb * 1048576) { showError("The file is larger than " + limitMb + " MB."); return; }
    url = "api/predict?" + scaleParam() + "&explain=1";
    options = { method: "POST", body: img.file,
                headers: Object.assign({ "Content-Type": img.file.type || "application/octet-stream" }, authHeaders()) };
  } else {
    url = "api/examples/" + encodeURIComponent(img.ex.id) + "/predict?explain=1";
    options = { method: "POST", headers: authHeaders() };
  }
  busy = true;
  mode = "busy";
  freeAnswer();
  answer = null;
  hideMap();
  $("stage").classList.add("analyzing");
  $("empty").hidden = true; $("out").hidden = true; $("chat").hidden = true;
  $("busy-text").textContent = "Analyzing " + img.name + "…";
  $("busy").hidden = false;
  $("result").setAttribute("aria-busy", "true");
  syncFoot();
  const steps = runSteps();
  const shortest = sleep(calm.matches ? 400 : MIN_MS);
  const sent = health && health.token_required ? $("token").value.trim() : null;   // the code this request carries
  let res = null, err = null;
  try {
    const r = await fetch(url, options);
    if (sent !== null && r.ok) rememberCode(sent);
    if (sent !== null && r.status === 401) rememberCode("");
    const body = await r.json().catch(() => ({}));
    if (r.status === 401) err = "Access code missing or wrong. Open the page with its link again, or type the code under the image.";
    else if (!r.ok) err = body.detail || "Server error (HTTP " + r.status + ").";
    else res = body;
  } catch (e) {
    err = "The server could not be reached.";
  }
  await shortest;
  await steps.finish(res);
  busy = false;
  $("stage").classList.remove("analyzing");
  $("result").setAttribute("aria-busy", "false");
  if (id !== seq) { syncFoot(); return; }   // the stage changed meanwhile: this answer belongs to an earlier image
  if (res) showAnswer(res); else showError(err);
}

// The model's own score for its answer, as a percentage with one decimal, never above 99.9: a network's output is not certainty.
function confidence(r) {
  const p = r.answer === "PTC-like" ? r.probability_ptc_like : 1 - r.probability_ptc_like;
  return Math.min(99.9, Math.round(p * 1000) / 10).toFixed(1) + "%";
}

function showAnswer(res) {
  answer = res;
  mode = "done";
  const ok = res.status === "ok";
  const sure = ok && res.answer !== "uncertain";               // a percentage only goes with a PTC or Non-PTC answer
  $("reasons").replaceChildren();
  $("note").hidden = true;
  if (ok) {
    const [label, cls] = VERDICT[res.answer] || [res.answer, "d-unc"];
    setVerdict(label, cls);
    if (sure) $("conf-num").textContent = confidence(res);
  } else {
    setVerdict("Refused", "d-ref");
    for (const r of res.refusal_reasons || []) $("reasons").append(el("li", r));
  }
  $("conf").hidden = !sure;
  const hm = res.heatmap;
  if (ok && hm && hm.made) showPair(hm);
  else if (ok) $("no-map").textContent = (hm && hm.reason) || "No heat map for this image.";
  $("busy").hidden = true; $("empty").hidden = true; $("out").hidden = false;
  showQuestions();
  syncFoot();
  setTimeout(reveal, oneColumn.matches && !calm.matches ? 950 : 0);   // on one column, once the two images are in place
}

function showError(msg) {
  answer = null;
  mode = current ? "ready" : "empty";
  setVerdict("Not analyzed", "d-ref");
  $("note").textContent = msg; $("note").hidden = false;
  $("conf").hidden = true;
  $("reasons").replaceChildren();
  $("busy").hidden = true; $("empty").hidden = true; $("out").hidden = false; $("chat").hidden = true;
  syncFoot();
  reveal();
}

// After the answer, two separate images take the stage's place: the selected image, and the same image with the Grad-CAM over it.
// The Grad-CAM is drawn from the same pixels, so the two line up; the grey strips on it were never read by the model.
function showPair(hm) {
  $("pair").style.setProperty("--ar", (hm.width / hm.height).toFixed(4));
  const orig = $("hm-scan");                                   // the selected image as it is (no dimmed strips); the scan the map was drawn on if it cannot be shown
  $("pair-scan").src = orig.complete && orig.naturalWidth > 0 ? orig.src : blobUrl(hm.scan_jpeg);
  $("pair-heat").src = blobUrl(hm.overlay_jpeg);
  $("stage-note").hidden = true;
  $("stage").hidden = true;
  $("pair").hidden = false;
  document.querySelector(".layout").classList.add("has-pair");
}

function showEmpty() {
  $("empty-lead").textContent = current ? current.name + " is ready. Press Analyze the image."
    : "No image analyzed yet. Upload one or pick a sample image below.";
  $("empty").hidden = false; $("busy").hidden = true; $("out").hidden = true; $("chat").hidden = true;
}

// On one column the result sits below the stage, and on wider screens the docked strip covers the bottom of the window: bring the
// answer into view when it arrives.
function reveal() {
  const dock = $("examples-card");
  const bottom = !dock.hidden && getComputedStyle(dock).position === "sticky" ? dock.getBoundingClientRect().top : window.innerHeight;
  const v = $("verdict").getBoundingClientRect();
  if (v.top < 0 || v.bottom > bottom) $("result").scrollIntoView({ behavior: calm.matches ? "auto" : "smooth", block: "start" });
}

// The moving cells of the analysis animation, spread over a 12 × 6 grid without repeats.
function initFx() {
  for (let i = 0; i < 18; i++) {
    const c = el("span", undefined, "fx-cell");
    c.style.left = ((i * 5) % 12) * (100 / 12) + "%";
    c.style.top = ((i * 7 + Math.floor(i / 12)) % 6) * (100 / 6) + "%";
    c.style.animationDelay = ((i * 0.37) % 1.8).toFixed(2) + "s";
    $("fx").append(c);
  }
}

// --- about this result ---------------------------------------------------------------------------------------------------------
// A short list of questions about this one image. Each opens to a prepared answer, filled in from this analysis (and, for a PTIC
// image, from the dataset's label). There is no text box and no chatbot; nothing leaves this server.

const NOT_A_DIAGNOSIS = "This tool describes how a patch of tissue looks, not a patient's diagnosis. It is for research use only and is not a medical device.";

const TOPICS = [
  { id: "meaning", q: "What does this result mean?", when: (r) => r.status === "ok",
    answer: (r) => {
      if (r.answer === "PTC-like") {
        return [`"Papillary thyroid carcinoma detected" (${PTC}) means the tissue in this image looks like papillary thyroid carcinoma to the model. Papillary thyroid carcinoma is the most common type of thyroid cancer.`, NOT_A_DIAGNOSIS];
      }
      if (r.answer === "not PTC-like") {
        return [`"No papillary thyroid carcinoma detected" (${NON_PTC}) means the tissue in this image does not look like papillary thyroid carcinoma to the model.`, NOT_A_DIAGNOSIS];
      }
      return ["The model is not clear enough about this image, so it gives neither of its two answers: papillary thyroid carcinoma detected, or no papillary thyroid carcinoma detected.", NOT_A_DIAGNOSIS];
    } },
  { id: "refused", q: "Why was it refused?", when: (r) => r.status !== "ok",
    answer: (r) => ["It did not pass the quality checks:",
                    { list: (r.refusal_reasons || []).length ? r.refusal_reasons : ["the image could not be read"] },
                    "The model only answers for images that look like its training scans; it refuses the others instead of guessing."] },
  { id: "sure", q: "How sure is the model?", when: (r) => r.status === "ok" && !!r.ensemble,
    answer: (r) => r.answer === "uncertain"
      ? ["The model is not sure about this image: it cannot choose between \"papillary thyroid carcinoma detected\" and \"no papillary thyroid carcinoma detected\"."]
      : [`The model's confidence in this answer is ${confidence(r)}.`,
         "That is how strongly the model favours its own answer. It is not the chance that a patient has cancer, and a model can be confident and still wrong."] },
  { id: "where", q: "Where did the model look?", when: (r) => r.status === "ok" && !!(r.heatmap && r.heatmap.made),
    answer: () => [`The Grad-CAM image colours each part of the picture by how much it pushed the model towards papillary thyroid carcinoma (${PTC}): blue means none, yellow to red means strong.`,
                   "It shows where, not why. It cannot rule out that the model reacts to the colour of its training hospital's stain or scanner instead of the cells."] },
  { id: "label", q: "Does it match the dataset label?", when: (r) => !!r.known_ptic_image,
    answer: (r) => {
      const k = r.known_ptic_image, lab = k.dataset_label === "PTC-like" ? "PTC-like" : "not PTC-like";   // the server's names, to compare with r.answer
      const out = [`This image comes from the PTIC dataset, where pathologists labelled it ${SHORT[lab]}.`];
      if (r.status !== "ok") { out.push("It was refused, so there is no answer to compare."); return out; }
      if (r.answer === "uncertain") out.push("The model is uncertain about it, so there is nothing to compare.");
      else if (r.answer === lab) out.push("The model agrees.");
      else out.push(`The model disagrees: it answers ${SHORT[r.answer] || r.answer}.` +(r.example && r.example.group === "Known hard case"
        ? " This patient's tissue is a known hard case for the model." : ""));
      out.push("The models that scored it never saw this image's patient, so this is a fair test.");
      return out;
    } },
  { id: "checks", q: "Did the image pass the quality checks?", when: (r) => Object.keys(r.checks || {}).length > 0,
    answer: (r) => {
      const all = Object.entries(r.checks), bad = all.filter(([, c]) => !c.passed);
      if (!bad.length) return [`Yes, all ${all.length} checks passed:`, { list: all.map(([n]) => NAMES[n] || n), cols: true }];
      return [`No, ${bad.length} of ${all.length} checks failed:`, { list: bad.map(([n, c]) => (NAMES[n] || n) + (c.reason ? " (" + c.reason + ")" : "")) }];
    } },
];

// One row per question that applies to this result, all closed. They share a name, so opening one closes the other (where the
// browser supports that; elsewhere several can stay open, which does no harm).
function showQuestions() {
  const box = $("qa");
  box.replaceChildren();
  for (const t of answer ? TOPICS.filter((t) => t.when(answer)) : []) {
    const row = el("details", undefined, "qa-item");
    row.setAttribute("name", "about-result");
    row.append(el("summary", t.q, "qa-q"));
    const body = el("div", undefined, "qa-a");
    for (const b of t.answer(answer)) {
      if (typeof b === "string") body.append(el("p", b));
      else if (b && b.list) { const ul = el("ul", undefined, b.cols ? "cols" : ""); for (const li of b.list) ul.append(el("li", li)); body.append(ul); }
    }
    row.append(body);
    row.addEventListener("toggle", () => { if (row.open) revealAnswer(row); });
    box.append(row);
  }
  $("chat").hidden = box.children.length === 0;
}

// An answer that opens under the docked strip is brought into view (the summary stays visible).
function revealAnswer(row) {
  const dock = $("examples-card");
  const floor = !dock.hidden && getComputedStyle(dock).position === "sticky" ? dock.getBoundingClientRect().top : window.innerHeight;
  const r = row.getBoundingClientRect();
  const by = Math.min(r.bottom - floor + 12, r.top - 8);
  if (by > 0) window.scrollBy({ top: by, behavior: calm.matches ? "auto" : "smooth" });
}

// --- light or dark: the system's choice until the visitor presses the button, then theirs (remembered in this browser) -------------
const THEME_KEY = "ptic-demo-theme";

function darkNow() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function initTheme() {
  const btn = $("theme");
  const show = () => btn.setAttribute("aria-pressed", darkNow() ? "true" : "false");
  btn.addEventListener("click", () => {
    const next = darkNow() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage blocked: not remembered */ }
    show();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", show);
  show();
}

document.addEventListener("DOMContentLoaded", () => {
  $("file").addEventListener("change", () => {
    const f = $("file").files[0];
    $("file").value = "";                   // so that choosing the same file again still counts
    if (f) loadUpload(f);
  });
  $("upload-btn").addEventListener("click", () => { if (!busy) $("file").click(); });
  document.querySelectorAll("input[name=scale]").forEach((r) => r.addEventListener("change", scaleChanged));
  $("mpp").addEventListener("input", scaleChanged);
  $("analyze").addEventListener("click", analyze);
  // A file dropped anywhere on the page must not replace it; one dropped on the image card goes on the stage.
  for (const t of ["dragover", "drop"]) window.addEventListener(t, (e) => e.preventDefault());
  const card = $("stage-card"), stage = $("stage");
  card.addEventListener("dragover", () => { if (!busy) stage.classList.add("over"); });
  card.addEventListener("dragleave", (e) => { if (!card.contains(e.relatedTarget)) stage.classList.remove("over"); });
  card.addEventListener("drop", (e) => {
    stage.classList.remove("over");
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) loadUpload(f);
  });
  const about = $("about");
  $("about-open").addEventListener("click", () => about.showModal());
  $("about-close").addEventListener("click", () => about.close());
  about.addEventListener("click", (e) => { if (e.target === about) about.close(); });   // a click on the backdrop
  $("refs").addEventListener("toggle", loadFigures);
  initFx();
  restoreCode();
  loadHealth();
  loadExamples();
  initStrip();
  initTheme();
  syncFoot();
});
