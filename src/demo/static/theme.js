"use strict";
// Runs before the page paints, so a theme picked in this browser never flashes: "light" or "dark", otherwise the system's.
try {
  const t = localStorage.getItem("ptic-demo-theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch (e) { /* storage blocked: follow the system */ }
