"use strict";
// Lets the unchanged demo page (src/demo/static) run inside a Streamlit component. The page asks its server for "api/..." with fetch()
// and <img src>; there is no web server behind a Streamlit app, so this script answers those requests through Streamlit's component
// channel: they go to the Python process (src/demo/rpc.py) and the replies come back as ordinary Response objects and blob: images.
// Requests are sent in batches, one batch at a time. Nothing else of the page is touched.
(() => {
  const REPLY_TIMEOUT_MS = 180000;                 // an analysis on a shared CPU takes up to about 30 s
  const LOAD = Math.random().toString(36).slice(2, 8);   // a reply meant for an earlier load of this frame is never taken for ours
  const origFetch = window.fetch.bind(window);
  const pending = new Map();                       // request id -> { resolve, timer }
  let queue = [], inflight = null, ready = false, nextId = 1, batchNo = 0, answered = false, noticeTimer = null;

  const post = (msg) => window.parent.postMessage(Object.assign({ isStreamlitMessage: true }, msg), "*");
  function toBase64(bytes) {
    let s = ""; const CH = 0x8000;
    for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    return btoa(s);
  }
  function fromBase64(s) {
    const bin = atob(s), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // After a long sleep the first reply waits for the model to load: say so instead of showing an empty page.
  function notice(show) {
    let el = document.getElementById("ptic-wake");
    if (!show) { if (el) el.remove(); return; }
    if (el) return;
    el = document.createElement("div");
    el.id = "ptic-wake"; el.setAttribute("role", "status");
    el.textContent = "Starting the model. After the app has been asleep this takes about a minute.";
    el.style.cssText = "position:fixed;left:50%;top:12px;transform:translateX(-50%);z-index:99;padding:10px 16px;background:#161616;color:#fff;" +
                       "font:14px/20px 'IBM Plex Sans',system-ui,sans-serif;box-shadow:0 2px 6px rgba(0,0,0,.3)";
    document.body.append(el);
  }

  function send() {
    if (inflight || !ready || !queue.length) return;
    const items = queue.splice(0);
    inflight = { batch: LOAD + "-" + (++batchNo), ids: items.map((i) => i.id) };
    post({ type: "streamlit:setComponentValue", dataType: "json", value: { batch: inflight.batch, items } });
  }

  function settle(reply) {
    if (!inflight || reply.batch !== inflight.batch) return;
    inflight = null; answered = true; clearTimeout(noticeTimer); notice(false);
    for (const it of reply.items) {
      const p = pending.get(it.id);
      if (!p) continue;
      pending.delete(it.id); clearTimeout(p.timer);
      p.resolve(new Response(fromBase64(it.body), { status: it.status, headers: { "Content-Type": it.type } }));
    }
    send();
  }

  function rpc(method, url, body) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      const timer = setTimeout(() => {              // no reply: fail like a network error, and let the next request through
        pending.delete(id); reject(new TypeError("no reply from the server"));
        if (inflight && inflight.ids.includes(id)) { inflight = null; send(); }
      }, REPLY_TIMEOUT_MS);
      pending.set(id, { resolve, timer });
      queue.push({ id, method, path: url, body: body ? toBase64(body) : null });
      if (!answered && !noticeTimer) noticeTimer = setTimeout(() => { if (!answered) notice(true); }, 5000);
      send();
    });
  }

  const isApi = (u) => typeof u === "string" && /^api\//.test(u);

  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (!isApi(url)) return origFetch(input, init);
    const o = init || {};
    const body = o.body != null ? new Uint8Array(await new Response(o.body).arrayBuffer()) : null;
    return rpc(String(o.method || "GET").toUpperCase(), url, body);
  };

  const srcDesc = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
  const blobs = new Map();                          // "api/..." -> Promise of a blob: URL
  Object.defineProperty(HTMLImageElement.prototype, "src", {
    configurable: true, enumerable: srcDesc.enumerable,
    get() { return srcDesc.get.call(this); },
    set(value) {
      if (!isApi(value)) { srcDesc.set.call(this, value); return; }
      if (!blobs.has(value)) blobs.set(value, rpc("GET", value).then((r) => r.ok ? r.blob() : Promise.reject(new Error("HTTP " + r.status))).then((b) => URL.createObjectURL(b)));
      blobs.get(value).then((u) => srcDesc.set.call(this, u)).catch(() => { blobs.delete(value); });
    },
  });

  // The frame fills the browser window, as the page does when it is opened on its own.
  function fit() {
    let h = 800;
    try { h = window.parent.innerHeight; } catch (e) { /* cross-origin parent: keep the default */ }
    post({ type: "streamlit:setFrameHeight", height: h });
  }
  try { window.parent.addEventListener("resize", fit); } catch (e) { /* see above */ }

  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || d.type !== "streamlit:render") return;
    if (!ready) { ready = true; fit(); send(); }
    const reply = d.args && d.args.reply;
    if (reply) settle(reply);
  });
  post({ type: "streamlit:componentReady", apiVersion: 1 });
})();
