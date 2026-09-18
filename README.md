# Paper2MD

Photograph the pages of a question paper, upload them, and get back **one
Markdown file** containing every question — formulas as LaTeX, every figure
cropped out and placed under the question it belongs to, and the student's
handwritten answers erased.

Built for maths/physics-style papers where the wording, the formulas *and* the
diagrams all matter.

```
photos of pages
      │
      ├─ PP-DocLayoutV3 (local, ~2.5 s) ──▶ pixel-accurate figure boxes ──▶ crop
      │
      └─ DeepSeek vision (~30 s) ─────────▶ markdown with LaTeX, handwriting already
                                            removed, and which question each
                                            figure belongs to
                                                     │
                                    figures cropped locally, matched to questions
                                                     │
                                    upload figures ──▶ imgbb.com (public URLs)
                                                     ▼
                                            ![Q17 附图](https://…)
```

---

## Quick start

```bash
./install.sh                 # full: local PaddleOCR-VL + optional DeepSeek
./install.sh --no-paddle     # DeepSeek only: ~30 MB of deps, ~300 MB of RAM

./run.sh                     # start
```

Then open **http://localhost:8000**.

The full install pulls PaddlePaddle (~200 MB) and, on the first page read, the
PaddleOCR-VL models (~2–4 GB). `--no-paddle` skips all of it and sets
`OCR_ENGINE=deepseek` for you — see [Engines](#engines). Pick that if you are
happy to depend on the API and want an install measured in megabytes rather
than gigabytes.

Either way the installer uses the Tsinghua PyPI mirror by default; override with
`PIP_INDEX=... ./install.sh`.

To try it immediately there is a ready-made exam page in
[`samples/exam_page.jpg`](samples/exam_page.jpg).

---

## Engines

Which model reads a page is the single biggest choice in the app.

```bash
OCR_ENGINE=auto      # auto | paddle | deepseek
```

| engine | how it works | time/page | memory | needs |
|---|---|---|---|---|
| `paddle` | PaddleOCR-VL end to end, all local | ~90–130 s | ~9 GB | nothing (offline) |
| `deepseek` | local layout boxes + DeepSeek reads the text | **~30 s** | ~300 MB | DeepSeek API key |
| `auto` *(default)* | `deepseek`, falling back to `paddle` if the API or layout model fails | — | — | key, but degrades safely |

**Why the hybrid, rather than letting DeepSeek do everything?** Because its
bounding boxes are only approximate. Measured against PaddleOCR-VL's on a real
page, its boxes disagreed on granularity — it split regions Paddle merged, and
flagged a tax table as a figure. Cropping from an approximate box either clips
the diagram or keeps the very handwriting you are trying to remove.

`PP-DocLayoutV3` — the *detection* half of PaddleOCR-VL, without the 0.9B
recognition model — gives pixel-accurate boxes in **~2.5 s, locally and free**.
So the pixels come from there and DeepSeek is asked only what it is good at:
reading the page, and saying which question each figure belongs to.

DeepSeek reads better than PaddleOCR-VL (cleaner LaTeX) **and removes the
handwriting in the same pass**, so the separate cleanup step is not needed. It
also has a useful side effect: layout detection separates figures Paddle merges
— a 480×173 region covering both a cylinder diagram and a bar chart came back as
two figures, correctly placed under different questions.

**Timing is dominated by reasoning, not by the image.** It varies a lot per page:

| case | reasoning tokens | time |
|---|---|---|
| fresh page, ordinary | ~4,500 | ~26 s |
| fresh page, heavy | ~13,000 | ~64 s |
| page re-run (prompt cached) | 235–312 | 7.6–9.3 s |

Budget **~30 s per page**, occasionally up to a minute. A *new* page is always a
cold call: DeepSeek caches the prompt, so a re-run skips the image prefill
(`prompt_cache_hit_tokens` → 1,152), but a fresh page is a fresh generation.

If the API is down, or you would rather stay fully offline, set
`OCR_ENGINE=paddle` and nothing else changes.

---

## Using it

1. **New task** — give the paper a title, then select every photo at once.
   *Upload order is page order*: the first file becomes page 1. Drag-and-drop
   works too.
2. **Wait for reading** — the status page polls itself and shows a progress bar
   (`n / total` pages). ~30 s per page with `auto`, 1–3 minutes with `paddle`.
3. **Review figures** — each figure gets a card. The engine's detected question
   numbers arrive **pre-filled**, so you are correcting rather than typing.
   * **Auto-fill: next questions** — numbers figures top-to-bottom from a
     question you pick (uses the recorded coordinates, so "top to bottom" is the
     real reading order).
   * **Auto-fill: match detected** — matches figures to the question numbers
     found on the page.
   * Or just type. Changes save automatically and the preview re-renders.
4. **Upload figures** — sends them to the image host and shows the exact final
   markdown, so links can be checked before downloading.
5. **Clean up** *(optional with `auto`)* — see [Cleanup](#cleanup).
6. **Download** — the whole paper as `.md`, or a single page. Once cleaned, a
   **Cleaned .md** button appears alongside; the original is always kept.

Both long-running steps have a progress bar: page reading, and the cleanup
(figures uploaded, then cleaned chunk by chunk).

### Figure format

Each figure is a markdown image whose alt text names its question, so the figure
and its question stay together in any renderer:

```markdown
![Q17 附图](https://i.ibb.co/93sZK3jf/a6283adda502.jpg)
![Q17 附图2](https://i.ibb.co/WqKW04v/d5ff1fbf1ec7.jpg)
```

A question with one figure gets `附图`; a question with several gets `附图1`,
`附图2`, so two diagrams under one question stay distinguishable.

### Cleanup

Both engines leave you with printed questions only, but they get there
differently. `deepseek` removes the handwriting as it reads. `paddle` cannot tell
handwriting from print, so it needs the separate pass:

> **✨ Clean up** sends the paper to DeepSeek and returns the printed questions
> with the handwriting turned back into blank slots — `则 $x+y=\underline{5}$`
> becomes `则 $x+y=\underline{\hspace{2em}}$`.

The DeepSeek pass reads the question number out of each image's alt text, swaps
every figure for an opaque `[[FIG3-Q13]]` marker ("figure 3 belongs to question
13"), and swaps them back afterwards — so the model can neither corrupt a URL
nor reattach a figure to the wrong question. Anything it drops is re-attached
rather than lost, and the task page reports how many were recovered.

Cleaned output is cached per task and **invalidated automatically** when a figure
mapping changes, a page is retried, or more pages are uploaded.

**Should the page photo be sent too?** Measured: no. `deepseek-flash` accepts
images, but on a handwriting-heavy page the image pass cost **2× the prompt
tokens and was less accurate** — handed
`（1）请用文字补全上述规律：<student's answer>`, text-only reduced it to a blank
while the image pass kept the answer as printed text. The text-only pass already
has the cue it needs. Reproduce with `tools/image_vs_text.py`.

**Truncated replies are rejected, not accepted.** A reply cut off at the model's
output limit returns HTTP 200 with `finish_reason: "length"`; that is now an
error with an actionable message rather than a silently incomplete paper.

### Handwriting that got extracted as a figure

PaddleOCR-VL occasionally cuts a region of **handwriting** out as if it were a
figure — a student's working, a tick, a circled option letter — and those end up
embedded in the paper. The cleanup cannot help: by then they are images, not text.

* **Delete one by hand** — every figure card has an **✕**. Pressing it marks the
  figure for deletion (the card greys out, so it is reversible) and keeps it out
  of the preview and every download.
* **Find them automatically** — **🔍 Detect handwriting** sends each figure to
  DeepSeek's vision model with one narrow question: printed material, or
  handwriting? Unassigned handwriting is marked for removal; a figure you already
  gave a question number is only flagged, never deleted.

That narrow question is where vision genuinely works — unlike the whole-page
cleanup, where it measured *worse*. On real crops: a printed cylinder and a
printed geometric figure were kept; a handwritten `B)` with a pen stroke and a
block of handwritten algebra were flagged. A region can contain **both** a
printed diagram and handwriting, which is exactly why the result is a suggestion
you can override rather than an automatic delete.

Both controls write `drop` as the figure's mapping value, which
`apply_labels_mapping()` acts on — the same place question numbers are applied,
so deleting and labelling compose.

---

## Configuration

Copy `.env.example` to `.env` (the installer does this).

### Engines and layout

| Variable | Default | Meaning |
|---|---|---|
| `OCR_ENGINE` | `auto` | `auto` / `paddle` / `deepseek` — see [Engines](#engines). |
| `LAYOUT_MODEL_NAME` | `PP-DocLayoutV3` | Layout model the `deepseek` engine uses for boxes. |
| `LAYOUT_MIN_SCORE` | `0.5` | Ignore detections below this confidence. |
| `LAYOUT_MIN_PIXELS` | `32` | A "figure" smaller than this is treated as a stray mark. |
| `LAYOUT_PADDING` | `6` | Pixels of margin around each crop, so nothing is clipped. |

### PaddleOCR-VL

| Variable | Default | Meaning |
|---|---|---|
| `OCR_MAX_DIM` | `1536` | Longest edge fed to the model. Lower is faster, less accurate. |
| `OCR_WORKERS` | `4` | Pages read at once. All share one model, so this is nearly free in RAM. |
| `OCR_THREADS` | `1` | Native threads per inference. 1 is fastest — see [Performance](#performance). |
| `OCR_ENABLE_MKLDNN` | `0` | oneDNN acceleration; measured *slower* here, leave off. |
| `OCR_PIPELINE_VERSION` | `v1.6` | PaddleOCR-VL pipeline version. |
| `OCR_DEBUG` | `0` | Print full tracebacks for failed pages. |

### DeepSeek

| Variable | Default | Meaning |
|---|---|---|
| `DEEPSEEK_API_KEY` | *(empty)* | Unset disables the engine *and* the cleanup button. |
| `DEEPSEEK_MODEL` | `deepseek-flash` | Verified against `GET /models` on first use. |
| `DEEPSEEK_FALLBACK_MODEL` | `deepseek-chat` | Used when the account lacks the above. |
| `DEEPSEEK_MAX_CHARS` | `24000` | Longer papers are cleaned chunk by chunk. |
| `DEEPSEEK_TIMEOUT` | `600` | Seconds per API call. |

### Where figures are hosted

| `IMAGE_HOST` | What it does | Trade-off |
|---|---|---|
| `imgbb` *(default)* | Uploads to [imgbb.com](https://api.imgbb.com/); markdown references the public URL | Needs a free `IMGBB_API_KEY`; URLs outlive the app |
| `local` | References `PUBLIC_BASE_URL/tasks/<id>/images/<file>` | Nothing leaves the machine, but links only work while the app runs |
| `base64` | Inlines every figure as a `data:` URI | One self-contained file, but large and not separately hostable |

Any figure that fails to upload is inlined as base64, so a download is never left
with a broken link.

With `IMAGE_HOST=local`, set `PUBLIC_BASE_URL` to the address other devices
should use (e.g. `http://192.168.1.20:8000`).

### Uploads

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` | `25` | Reject photos larger than this. |
| `MAX_PAGES_PER_UPLOAD` | `50` | Files per request; the browser batches larger selections. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Where the app listens. |

---

## How it works

One process — no second service to keep alive.

```
app/
  main.py            FastAPI routes + Jinja templates
  worker.py          OCR queue, worker threads, and the cleanup runner
  ocr.py             engine dispatch: PaddleOCR-VL, or the hybrid
  layout.py          PP-DocLayoutV3 wrapper — the figure boxes, ~2.5 s locally
  deepseek.py        page reading, cleanup, and figure classification
  markdown_utils.py  normalisation, figure refs, question labelling, merging
  image_host.py      imgbb / local / base64 hosting with a per-task URL cache
  store.py           SQLite: tasks, pages, mappings, cached URLs, cleaned text
templates/           index (upload), task (progress), edit (figure → question)
static/              app.css, app.js, vendored marked + KaTeX (works offline)
tools/               benchmarks, the DeepSeek probes, comparison helpers
data/                runtime: paper2md.db, uploads/<task>/{pages,images}
```

**Flow.** Uploading a page writes a JPEG to `data/uploads/<task>/pages/` and
queues a job. A worker thread resizes the photo (honouring EXIF rotation), runs
the configured engine, copies cropped figures into `images/`, and stores the
page's markdown. Task status is derived from its pages, so an interrupted run
resumes: pages left mid-flight are re-queued on the next start.

**Question numbers are data, not text.** They live in a per-page mapping
(`{filename: "13"}`) and are applied when markdown is *rendered*, never written
into the stored page text. That is what lets the output format change, lets the
review screen override an engine's guess, and lets a figure be re-labelled
without re-running OCR. Pages written by older versions, which did bake labels
into the text, are migrated at startup — the labels are recovered into the
mapping and stripped. A mapping you set is never overwritten.

**Figure filenames** carry their page and box
(`p2_img_in_image_box_X1_Y1_X2_Y2.jpg`) so two pages cannot collide and the
coordinates remain available for the auto-fill sort. The order is
`X1_Y1_X2_Y2`, verified against the saved figure pixels (correlating +0.99
against the correct reading, +0.04 against the swapped one).

**Downloads** rebuild markdown from each page plus its mapping, wrap pages in
`<!-- page N -->` markers, and rewrite figures to the configured host.

### HTTP surface

**Pages** (HTML): `/` upload + task list · `/tasks/{id}` progress · `/tasks/{id}/edit` figure review.

**JSON API:**

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/tasks` | Create a task |
| `GET` | `/api/tasks` | List tasks |
| `DELETE` | `/api/tasks/{id}` | Delete a task and its files |
| `POST` | `/api/tasks/{id}/pages` | Upload page photos (order = page order) |
| `POST` | `/api/tasks/{id}/pages/{n}/retry` | Re-run a failed page |
| `GET` | `/api/tasks/{id}` | Status, per-page state, cleanup progress |
| `GET` | `/api/tasks/{id}/pages/{n}` | Page markdown, mapping, figures, questions |
| `PUT` | `/api/tasks/{id}/pages/{n}/mapping` | Save `{filename: "13"}` (or `"drop"`) |
| `POST` | `/api/tasks/{id}/pages/{n}/classify-figures` | Flag figures that are handwriting |
| `POST` | `/api/tasks/{id}/host-images` | Upload figures, return hosted markdown |
| `POST` | `/api/tasks/{id}/polish` | Start the cleanup — `202`, then poll |
| `GET` | `/api/tasks/{id}/polish` | Cleanup progress and result |
| `GET` | `/api/tasks/{id}/markdown` | Merged markdown (`?variant=polished`) |
| `GET` | `/tasks/{id}/download` | Final `.md` (`?variant=polished`) |
| `GET` | `/tasks/{id}/pages/{n}/raw` | The original page photo |
| `GET` | `/tasks/{id}/images/{file}` | One figure |
| `GET` | `/api/health` | Status, engine, configured hosts |

---

## Performance

CPU inference badly under-uses a modern machine: a single PaddleOCR-VL page
occupies roughly **one core**. Running several pages at once against one shared
model is therefore close to a free speed-up.

Measured on a 16-core Intel Core Ultra 9 285H, `OCR_MAX_DIM=1024`:

| `OCR_WORKERS` | 3 pages | throughput | peak RAM |
|---|---|---|---|
| 1 | 266.8 s | 12.3 chars/s | 9.2 GB |
| 3 | 115.3 s | 28.5 chars/s | 9.2 GB |
| 6 | 171.0 s | 38.5 chars/s | 9.2 GB |

Note the flat memory: every worker shares one model instance, so raising
`OCR_WORKERS` costs almost no RAM. Scaling flattens past ~4 because the work
becomes memory-bandwidth bound. Start at about a third of your core count.

Lowering `OCR_MAX_DIM` from 1536 to 1024 is a further **1.7×** at a small cost in
detail.

`tools/bench.py` and `tools/concurrency.py` reproduce these on your machine.

`OCR_ENABLE_MKLDNN=1` was measured **slower** (187.6 s vs 172.7 s) on this CPU,
despite the reference project enabling it — hence the off default.

---

## Deployment

### Could this run on Vercel?

Not with PaddleOCR-VL, but the DeepSeek engine alone clears both hard blockers:

| | with Paddle | DeepSeek only | Vercel limit |
|---|---|---|---|
| function bundle | ~1.6 GB | **29.6 MB** | 250 MB unzipped |
| peak memory | 9.2 GB | ~300 MB | 1 GB Hobby / 3 GB Pro |
| time per page | 90–130 s | ~30 s | 60 s Hobby / 300 s Pro |

29.6 MB leaves 8× headroom, and ~30 s fits Hobby's ceiling **if you process one
page per request**.

What would still have to change is everything assuming a long-lived process:

1. **Storage** — SQLite and `data/uploads/` must become Postgres plus blob
   storage; the filesystem is read-only except `/tmp`. This is the bulk of the
   work.
2. **No background worker** — the queue and worker threads do not survive
   serverless; each request would process its own page synchronously.
3. **Figure boxes** — dropping Paddle loses PP-DocLayoutV3, so crops would come
   from DeepSeek's approximate boxes. A hosted layout API would be needed to keep
   current quality.
4. **No offline path** — every page would depend on the API.

Roughly a day's work, mostly storage. Worth weighing against a small container
host (Fly.io, Railway, Render, a VPS): with Paddle gone the app needs only
~300 MB, so a cheap instance runs the **same code unchanged** and keeps precise
figure crops.

### Why not the GPU?

The machine this was built on has an **Intel Arc 140T integrated GPU** and no
NVIDIA GPU, under WSL2:

* WSL2 exposes `/dev/dxg` (a D3D12 path, which is how CUDA reaches a GPU from
  WSL) but there is no `/dev/dri`, so the Linux Intel compute stack — Level Zero
  / oneAPI / OpenVINO — cannot see the iGPU at all.
* PaddlePaddle has no Intel GPU wheel; its "XPU" support means Baidu's Kunlun
  chips, not Intel Arc.
* An integrated Arc shares system RAM with the CPU, so it offers little extra
  memory bandwidth — the actual bottleneck.

It is not impossible: `PaddleOCRVL` accepts `vl_rec_backend="vllm-server"` plus
`vl_rec_server_url`, so the VLM could be served by an Intel-XPU build of vLLM in
Docker while layout detection stays local. That is a substantial project with an
uncertain payoff on an integrated GPU.

---

## Tests

```bash
.venv/bin/python tests/run.py        # no pytest needed
.venv/bin/pytest tests -q            # if you have pytest
```

78 tests. The OCR engine, the layout model and the DeepSeek API are all stubbed,
so the suite runs offline and fast, while the real app, database, worker threads,
markdown pipeline, cropping and box matching are exercised for real.

`tests/_env.py` points every test at a throwaway data directory **before
`app.config` is imported** — that ordering matters, because config reads its
settings once at import. A guard test fails loudly if it is ever bypassed, so the
suite cannot quietly run against your real database.

---

## Troubleshooting

**Everything is slow** — check which engine is running: `GET /api/health`. If it
says `paddle`, set `OCR_ENGINE=auto` (needs a DeepSeek key) for ~30 s pages.
Otherwise lower `OCR_MAX_DIM`.

**A page failed** — click **Retry** on its card; other pages are unaffected. Set
`OCR_DEBUG=1` and check the server log.

**A figure is missing from the paper** — its file was not on disk. The engine
logs `referenced figure not found on disk: <name>`; **Retry** the page.

**Cleanup says the model hit its output limit** — lower `DEEPSEEK_MAX_CHARS` so
the document is split into smaller chunks.

**Import error mentioning `libGL.so.1`** — PaddleOCR pulls in OpenCV, which wants
the GL runtime even headless:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

**Clean up button stays grey** — it should not; the button state is derived from
the polled job status on every tick. If it happens, check `GET
/api/tasks/<id>/polish` and the server log.

---

## Notes on the reference project

This app replaces `math-exam-parser` (at
`D:\Family\app\questionExtract\app\math-exam-parser`), reusing its proven parts
and fixing several things along the way:

* **One service instead of two.** The reference split OCR and the web UI into two
  FastAPI processes with polling and a 5-minute result TTL between them. Here the
  queue lives in the web process: one thing to start, and no window in which a
  slow page's result expires before it is collected.
* **A figure-box bug inherited from it.** The reference read
  `img_in_image_box_A_B_C_D` as if `A` were `y1`; it is `x1`. That made
  "Auto-fill: next questions" sort figures left-to-right instead of
  top-to-bottom. Verified against the saved pixels and fixed here.
* **Figure filename collisions.** The reference stored every page's figures in one
  shared folder, so identically-named figures from different pages overwrote each
  other. Figures are now prefixed with their page number.
* **Unmapped figures stay put.** The reference moved *every* figure to the end of
  the document until it was mapped; now only mapped figures move.
* **ImgBB instead of Imgur.** The reference's CLI used Imgur, whose anonymous
  uploads are heavily rate-limited (it returns HTTP 429 today); its web UI had
  already moved to ImgBB, which is what this app uses.
* **EXIF rotation** is applied to phone photos, and images are normalised before
  reaching a model.
* **Vendored preview libraries** (marked + KaTeX) so the review screen renders
  formulas without a CDN at runtime.
