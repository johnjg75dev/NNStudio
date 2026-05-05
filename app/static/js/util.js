/* Tiny utility helpers shared by every page. */
(function () {
  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) node.setAttribute(k, v);
    }
    for (const c of children) {
      if (c == null || c === false) continue;
      node.append(c.nodeType ? c : document.createTextNode(c));
    }
    return node;
  };

  const fetchJSON = async (url, opts = {}) => {
    const res = await fetch(url, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "Accept": "application/json", ...(opts.headers || {}) },
      ...opts,
      body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
    });
    const ct = res.headers.get("content-type") || "";
    const data = ct.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) {
      const msg = (data && data.error) || res.statusText;
      throw new Error(`${res.status} · ${msg}`);
    }
    return data;
  };

  const debounce = (fn, ms = 200) => {
    let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  };

  const fmtBytes = (n) => {
    if (n == null || isNaN(n)) return "—";
    const u = ["B", "KB", "MB", "GB"]; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 ? 2 : 1)} ${u[i]}`;
  };

  const fmtNum = (n, d = 4) => {
    if (n == null || isNaN(n)) return "—";
    const a = Math.abs(n);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return n.toExponential(2);
    return n.toFixed(d);
  };

  const timeAgo = (iso) => {
    if (!iso) return "—";
    const t = new Date(iso).getTime();
    const s = Math.max(1, Math.floor((Date.now() - t) / 1000));
    if (s < 60) return `${s}s ago`;
    const m = Math.floor(s / 60); if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60); if (h < 24) return `${h}h ago`;
    const d = Math.floor(h / 24); if (d < 30) return `${d}d ago`;
    return new Date(iso).toLocaleDateString();
  };

  const toast = (msg, kind = "ok", ms = 2600) => {
    let host = document.querySelector(".toast-host");
    if (!host) { host = el("div", { class: "toast-host" }); document.body.appendChild(host); }
    const t = el("div", { class: `toast ${kind}` }, msg);
    host.appendChild(t);
    setTimeout(() => { t.style.opacity = 0; t.style.transform = "translateY(8px)"; }, ms - 300);
    setTimeout(() => t.remove(), ms);
  };

  const modal = (title, bodyHTML, footerHTML = "") => {
    let host = document.querySelector(".modal-host");
    if (!host) { host = el("div", { class: "modal-host" }); document.body.appendChild(host); }
    host.innerHTML = "";
    const m = el("div", { class: "modal" });
    m.innerHTML = `
      <div class="modal-head"><h3>${title}</h3><button class="btn icon ghost" data-close>${ICONS.x()}</button></div>
      <div class="modal-body">${bodyHTML}</div>
      ${footerHTML ? `<div class="modal-foot">${footerHTML}</div>` : ""}
    `;
    host.appendChild(m);
    host.classList.add("open");
    const close = () => { host.classList.remove("open"); setTimeout(() => (host.innerHTML = ""), 180); };
    m.querySelector("[data-close]").addEventListener("click", close);
    host.addEventListener("click", (e) => { if (e.target === host) close(); });
    return { el: m, close };
  };

  window.NN = { $, $$, el, fetchJSON, debounce, fmtBytes, fmtNum, timeAgo, toast, modal };
})();
