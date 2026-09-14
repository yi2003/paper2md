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
