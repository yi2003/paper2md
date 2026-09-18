/* Paper2MD — shared front-end helpers. */

/* marked + KaTeX are vendored under /static/vendor so the app works offline. */
if (window.marked) {
  marked.use({ breaks: true, gfm: true });
}

const MATH_DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "$", right: "$", display: false },
  { left: "\\(", right: "\\)", display: false },
];

/** Render markdown (with LaTeX) into an element. */
function renderMarkdown(element, markdown) {
  if (!element) return;
  if (!window.marked) {
    element.textContent = markdown || "";
    return;
  }
  element.innerHTML = marked.parse(markdown || "");
  if (window.renderMathInElement) {
    try {
      renderMathInElement(element, {
        delimiters: MATH_DELIMITERS,
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "option", "pre"],
      });
    } catch (error) {
      console.warn("KaTeX render failed:", error);
    }
  }
}

/** fetch() + JSON + a useful error message. */
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const raw = await response.text();
  let data = null;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      data = { detail: raw };
    }
  }
  if (!response.ok) {
    const detail = data && (data.detail || data.error);
    throw new Error(typeof detail === "string" ? detail : `HTTP ${response.status}`);
  }
  return data;
}

/** POST/PUT a JSON body. */
function jsonRequest(method, body) {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  };
}

let toastTimer = null;
function toast(message, isError = false) {
  let element = document.querySelector(".toast");
  if (!element) {
    element = document.createElement("div");
    element.className = "toast";
    document.body.appendChild(element);
  }
  element.textContent = message;
  element.classList.toggle("err", !!isError);
  element.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), isError ? 5000 : 2200);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

function badgeClass(status) {
  return ["queued", "processing", "done", "partial", "failed"].includes(status)
    ? status
    : "queued";
}

function formatTime(unixSeconds) {
  if (!unixSeconds) return "";
  const date = new Date(unixSeconds * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
         `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** Download a URL as a file without leaving the page. */
function triggerDownload(url) {
  const frame = document.createElement("iframe");
  frame.style.display = "none";
  frame.src = url;
  document.body.appendChild(frame);
  setTimeout(() => frame.remove(), 60000);
}

/* ---------------------------------------------------------------- progress */

const STAGE_LABELS = {
  hosting: "Uploading figures",
  cleaning: "Cleaning with DeepSeek",
  done: "Finished",
};

/**
 * Draw a progress bar.
 *
 * @param {HTMLElement} container element to fill
 * @param {object} state {status, stage, done, total, error}
 * @param {object} [opts] {title, detail, hideWhenDone}
 */
function updateProgress(container, state, opts = {}) {
  if (!container) return;

  const idle = !state || state.status === "idle" || state.status === undefined;
  if (idle || (state.status === "done" && opts.hideWhenDone)) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }

  const running = state.status === "running";
  const failed = state.status === "failed";
  const finished = state.status === "done";
  const total = Number(state.total) || 0;
  // `done` is fractional while a chunk streams, so even a single chunk gives a
  // meaningful percentage. Only the "N / M" label needs more than one step.
  const hasCount = total > 0;
  const percent = finished
    ? 100
    : hasCount
      ? Math.min(100, Math.round((Number(state.done) / total) * 100))
      : 0;

  // No total at all (e.g. the first figure upload) — sweep rather than fake it.
  const indeterminate = running && !hasCount;
  const barClass = finished ? "done" : failed ? "failed" : indeterminate ? "indeterminate" : "";

  const title = opts.title || STAGE_LABELS[state.stage] || "Working…";
  const detail = opts.detail !== undefined
    ? opts.detail
    : total > 1
      // `done` can be fractional mid-chunk; show whole steps only.
      ? `${Math.floor(Number(state.done) || 0)} / ${state.total}`
      : "";

  container.hidden = false;
  container.innerHTML =
    '<div class="progress-wrap">' +
      '<div class="head">' +
        `<span class="title">${escapeHtml(title)}</span>` +
        `<span class="detail">${escapeHtml(detail || "")}</span>` +
        `<span class="pct">${failed ? "" : percent + "%"}</span>` +
      "</div>" +
      `<div class="progress ${barClass}"><div class="bar" style="width:${percent}%"></div></div>` +
      (state.error ? `<div class="progress-error">${escapeHtml(state.error)}</div>` : "") +
    "</div>";
}
