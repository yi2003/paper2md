# Paper2MD

Photograph the pages of a question paper, upload them, and get back **one
Markdown file** containing every question — with formulas as LaTeX and figures
extracted, labelled with the question they belong to, and hosted on the web.

Built for math/physics-style papers where the wording, the formulas *and* the
diagrams all matter.

```
photos of pages ──▶ PaddleOCR-VL ──▶ per-page markdown + figures
                                            │
                            label each figure with its question number
                                            │
                            upload figures ──▶ imgbb.com (public URLs)
                                            │
                                            ▼
                                    exam_paper.md
```

---

## Quick start

```bash
./install.sh     # creates .venv, installs PaddleOCR-VL (CPU)
./run.sh         # starts the app
```

Then open **http://localhost:8000**.

`install.sh` needs to download PaddlePaddle (~200 MB) and, on the first OCR run,
the PaddleOCR-VL models (~2–4 GB). It uses the Tsinghua PyPI mirror by default;
override with `PIP_INDEX=... ./install.sh`.

To try it immediately there is a ready-made exam page in
[`samples/exam_page.jpg`](samples/exam_page.jpg).

---

## Using it

1. **New task** — give the paper a title, then select every photo at once.
   *Upload order is page order*: the first file becomes page 1. Drag-and-drop
   works too.
2. **Wait for OCR** — the status page polls itself. On CPU expect roughly
   1–3 minutes per page; pages are processed one at a time by default.
3. **Review figures** — for each page, every extracted figure gets a card.
   * Auto-fill: next questions**: numbers the figures top-to-bottom starting
     from a question you pick (uses the coordinates PaddleOCR recorded, so
     "top to bottom" is the real reading order).
   * Auto-fill: match detected**: matches the figures to the question numbers
     found on the page.
   * Or just type the number next to each figure. Changes save automatically,
     and the preview re-renders with the figure sitting under its question.
4. **Upload figures** — sends the figures to the image host and shows the exact
   final markdown, so you can check the links before downloading.
5. **Clean up** *(optional)* — photos of a used paper contain the student's
   handwritten answers, and the OCR reads them as if they were part of the text
   (`则 $x+y=\underline{5}$`, `圆心角是 120 度`). With a DeepSeek key configured,
   **✨ Clean up** sends the paper to DeepSeek and returns the printed questions
   only, with the handwriting turned back into blank answer slots — see
   [DeepSeek cleanup](#deepseek-cleanup).
6. **Download** — the whole paper as `.md`, or a single page. Once cleaned, a
   **Cleaned .md** button appears alongside it; the original is always kept.

Both long-running steps show a progress bar: page reading (`n / total` pages)
and the DeepSeek cleanup (figures uploaded, then cleaning).

Each figure goes into the markdown as an image whose alt text names the question
it belongs to, so the figure and its question stay together in any renderer:

```markdown
![Q17 附图](https://i.ibb.co/93sZK3jf/a6283adda502.jpg)
![Q17 附图2](https://i.ibb.co/WqKW04v/d5ff1fbf1ec7.jpg)
```

A question with a single figure gets `附图`; a question with several numbers them
`附图1`, `附图2`, so two diagrams under one question are still distinguishable.

---

## Configuration

Copy `.env.example` to `.env` (the installer does this for you).

### Where figures are hosted

| `IMAGE_HOST` | What it does | Trade-off |
|---|---|---|
| `imgbb` *(default)* | Uploads every figure to [imgbb.com](https://api.imgbb.com/), markdown references the public URL | Needs a free `IMGBB_API_KEY`; URLs keep working after the app stops |
| `local` | Markdown references `PUBLIC_BASE_URL/tasks/<id>/images/<file>` | Nothing leaves your machine, but links only work while the app runs |
| `base64` | Inlines every figure as a `data:` URI | One fully self-contained file, but large and not separately hostable |

Any figure that fails to upload is automatically inlined as base64, so a
download is never left with a broken link.

```bash
IMGBB_API_KEY=your_key_here      # https://api.imgbb.com/
```

With `IMAGE_HOST=local`, set `PUBLIC_BASE_URL` to the address other devices
should use (e.g. `http://192.168.1.20:8000`) — `localhost` only resolves on the
machine running the app.

### OCR

| Variable | Default | Meaning |
|---|---|---|
| `OCR_MAX_DIM` | `1536` | Longest edge fed to the model. Lower is faster and less accurate. |
| `OCR_WORKERS` | `4` | Pages OCR'd at once. All share one model, so this is nearly free in RAM. |
| `OCR_THREADS` | `1` | Native threads per inference. 1 is fastest here — see below. |
| `OCR_ENABLE_MKLDNN` | `0` | oneDNN acceleration; measured *slower* here, leave off. |
| `OCR_PIPELINE_VERSION` | `v1.6` | PaddleOCR-VL pipeline version. |
| `OCR_DEBUG` | `0` | Print full tracebacks for failed pages. |

### DeepSeek cleanup

An optional final pass that turns raw OCR into a blank, question-only paper.

```bash
DEEPSEEK_API_KEY=sk-...        # https://platform.deepseek.com/
DEEPSEEK_MODEL=deepseek-flash  # checked against GET /models on first use
```

**Removes** — answers written into blanks, filled-in values, circled options,
working and margin notes. Blanks come back as `\underline{\hspace{2em}}`, so the
paper is answerable again.

**Keeps** — printed question stems, options, section headings, marks, all LaTeX,
and every figure. Question numbering is never changed.

**How the figures survive an LLM.** Asking a chat model to re-emit
`<img src="https://i.ibb.co/...">` is asking for a corrupted URL. So each figure
is swapped for an opaque marker before the call — `[[FIG3-Q13]]`, "figure 3
belongs to question 13" — and swapped back afterwards. The label is read out of
the image's alt text and folded into the marker, so the model cannot reattach a
figure to the wrong question. Anything the model drops is re-attached at the end rather than lost,
and the task page reports how many were recovered.

On a real 4-page paper the cleanup removed **30% of the text** (all handwriting)
while keeping **24 of 24 questions and 7 of 7 figures**.

| Variable | Default | Meaning |
|---|---|---|
| `DEEPSEEK_API_KEY` | *(empty)* | Unset disables the feature and hides the button. |
| `DEEPSEEK_MODEL` | `deepseek-flash` | Falls back to `DEEPSEEK_FALLBACK_MODEL` when unavailable. |
| `DEEPSEEK_MAX_CHARS` | `24000` | Longer papers are cleaned page by page. |
| `DEEPSEEK_TIMEOUT` | `600` | Seconds per API call. |

Cleaned output is cached per task and **invalidated automatically** whenever a
figure mapping changes, a page is retried, or more pages are uploaded.

Best results come from setting the figure → question mapping **before** cleaning,
so each figure carries its question number into the cleanup.

**Progress.** The cleanup runs in the background. `POST /api/tasks/<id>/polish`
returns `202` immediately and the UI polls `GET /api/tasks/<id>/polish`, so a
long run never looks like a hung request. The bar has two phases:

1. **Uploading figures** — a real `n / total` count, one step per figure.
2. **Cleaning with DeepSeek** — driven by the characters the model streams back.

`deepseek-flash` is a *reasoning* model: it can think for a minute or more
before emitting a single word of the answer (a typical page spent ~90 s thinking
and ~5 s answering, at 26k completion tokens for 3.4k characters of output).
Progress therefore counts the streamed `reasoning_content` as well as `content`,
and maps it through a soft curve that approaches — but never claims — completion.
Without that, the bar would sit at 0% for the whole thinking phase.

**Should the page photo be sent to DeepSeek too?** Measured: no. `deepseek-flash`
does accept images (it read a test number out of one), but on a handwriting-heavy
page the image pass cost **2x the prompt tokens** and was *less* accurate:

```
text-only  : （1）请用文字补全上述规律：\underline{\hspace{2em}}
text+image : （1）请用文字补全上述规律：把一个两位数的十位数字和个位数字交换位置，原来两位数与新的两位数的差是\underline{\hspace{2em}}
```

The student's answer was correctly removed in the first case and left in as if it
were printed in the second. The text-only pass already has the cue it needs —
text sitting after a "complete the following:" colon is almost certainly the
answer — and PaddleOCR-VL has already done the visual work at full resolution
(3072x4096), while an image attached to the cleanup call has to be downscaled to
roughly 1050x1400 to keep the request reasonable.

Re-run it on your own pages with `tools/image_vs_text.py <task_id> <page_index>`,
which does both passes and diffs them. Two pages is a small sample: on a page
where the OCR text is badly garbled, the photo may well help.

### Handwriting that got extracted as a figure

PaddleOCR-VL occasionally cuts a region of **handwriting** out as if it were a
figure — a student's working, a tick, a circled option letter — and those end up
embedded in the finished paper. The cleanup pass cannot help with these: by then
they are images, not text. Two controls on the review screen deal with it.

**Delete one by hand.** Every figure card has an **✕** button. Pressing it marks
that figure for deletion (the card greys out so the decision is reversible) and
keeps it out of the preview and every download. Under the hood the mapping value
becomes `drop`, which `apply_labels_mapping()` acts on — the same place question
numbers are applied, so deleting and labelling compose naturally.

**Find them automatically.** **🔍 Detect handwriting** sends each figure on the
page to DeepSeek's vision model and asks one narrow question: printed material, or
handwriting? Cards get labelled `printed` or `looks handwritten`, and the
unassigned handwriting is marked for removal. A figure you have already given a
question number is only flagged, never deleted.

That narrow question is where vision genuinely works well — unlike the whole-page
cleanup, where it measured *worse*. On real crops from these papers:

| figure | classified | correct? |
|---|---|---|
| printed cylinder diagram (65×90) | `printed` | ✅ |
| printed geometric figure (65×43) | `printed` | ✅ |
| handwritten "B)" with pen stroke (112×64) | `handwritten` | ✅ |
| handwritten algebra over a printed diagram (492×150) | `handwritten` | ✅ |

Note the last row: a region can contain **both** a printed diagram and
handwriting. The classifier makes a judgement call, which is exactly why the
result is a suggestion you can override with **✕** rather than an automatic
delete.

On one real page all 6 extracted "figures" were handwriting; marking them
removed every one, taking the downloaded paper from 5,696 to 4,997 characters
with no false positives.

### Which engine reads a page

```bash
OCR_ENGINE=auto      # auto | paddle | deepseek
```

| engine | how it works | time/page | needs |
|---|---|---|---|
| `paddle` | PaddleOCR-VL end to end | ~90–130 s | nothing (local, offline) |
| `deepseek` | local layout detection for the figure boxes + DeepSeek vision for the text | **~30 s** | DeepSeek API key |
| `auto` | `deepseek`, falling back to `paddle` if the API or layout model fails | — | key, but degrades safely |

**`auto` is the default.** It measured **~4× faster** than `paddle` with better
output, and falls back rather than failing if the API is unreachable.

Timing varies a lot with how much the model reasons before answering, and
reasoning is the dominant cost — it is not the image upload. Measured:

| case | reasoning tokens | time |
|---|---|---|
| fresh page, heavy reasoning | 4,563 | 25.6 s |
| fresh page, very heavy reasoning | ~13,000 | ~64 s |
| page already processed (re-run) | 235–312 | 7.6–9.3 s |

DeepSeek caches the prompt, so a re-run skips the image prefill (`prompt_cache_hit_tokens`
goes to 1,152) — but a **new** page is always a cold call and a fresh generation.
Budget **~30 s per page**, with occasional runs up to a minute.

Why the split, rather than letting DeepSeek do everything? Because its bounding
boxes are only approximate. Measured against PaddleOCR-VL's on a real page, its
boxes disagreed on granularity — it split regions Paddle merged, and flagged a
tax table as a figure. Cropping from an approximate box risks clipping the
diagram or keeping the very handwriting you are removing.

`PP-DocLayoutV3` — the detection half of PaddleOCR-VL, without the 0.9B
recognition model — gives pixel-accurate boxes in **~2.5 s locally, for free**.
So:

```
page photo
   ├─ PP-DocLayoutV3 (local, 2.5 s)  -> pixel-accurate figure boxes -> crop
   └─ DeepSeek vision (~5 s)         -> markdown with LaTeX, handwriting already removed
                                          + which question each figure belongs to
   -> boxes cropped locally, matched to those hints, placed under the question
```

DeepSeek is asked only *which question each figure belongs to* — a judgement it
is good at — while the pixels come from the local model.

Measured on a real page: all 12 questions present, tax table intact, and the
handwriting (`5`, `120`, `157`, `24元`) removed in the same pass — no separate
cleanup step needed.

Two bonus effects worth knowing:

- **It splits figures PaddleOCR merges.** On the sample page PaddleOCR returned
  one 480×173 region covering both a cylinder diagram and a bar chart; layout
  detection returns them as two separate figures.
- **The DeepSeek cleanup pass becomes optional.** The text already comes back
  clean, so **✨ Clean up** is only needed if you want a second pass.

Figure placement uses a 0.3 overlap threshold: a weak match leaves the figure
unplaced (it appears at the end) rather than attaching it to the wrong question,
and the review screen can fix it by hand.

If the API is down or you would rather stay fully offline, set `OCR_ENGINE=paddle`
and nothing else changes.

### Could this run on Vercel?

Not with PaddleOCR-VL — it needs 1.9 GB of models and 9.2 GB of RAM, against
Vercel's 250 MB bundle and 1 GB (Hobby) / 3 GB (Pro) memory limits. But **with
the DeepSeek engine only**, both of those blockers disappear:

| | with Paddle | DeepSeek only | Vercel limit |
|---|---|---|---|
| function bundle | ~1.6 GB | **29.6 MB** | 250 MB |
| peak memory | 9.2 GB | ~300 MB | 1 GB / 3 GB |
| time per page | 90–130 s | ~30 s | 60 s Hobby / 300 s Pro |

29.6 MB leaves 8× headroom, and ~30 s fits inside Hobby's 60 s ceiling if you
process **one page per request**.

What would still have to change is everything that assumed a long-lived process:

1. **Storage.** SQLite and `data/uploads/` have to move to Postgres plus blob
   storage (Vercel Blob, R2, S3) — the filesystem is read-only except `/tmp`.
   `store.py` and the image paths are the bulk of the work.
2. **No background worker.** The OCR queue and worker threads do not survive
   serverless. Each upload request would process its own page synchronously.
3. **Figure boxes.** Dropping Paddle means losing PP-DocLayoutV3, so crops would
   come from DeepSeek's own approximate boxes — noticeably less precise. A
   hosted layout API would be needed to keep the current quality.
4. **No offline path.** Every page would depend on the API.

Realistically about a day's work, mostly the storage layer. Worth weighing
against just running the same app on a small container host (Fly.io, Railway,
Render, a VPS): with Paddle gone it needs only ~300 MB of RAM, so even a cheap
instance would do, and the code runs unchanged.

### Performance

CPU inference is slow and, by default, badly under-uses a modern machine: a
single page occupies roughly **one core**. Running several pages at once against
one shared model is therefore close to a free speed-up.

Measured on a 16-core Intel Core Ultra 9 285H, `OCR_MAX_DIM=1024`:

| `OCR_WORKERS` | 3 pages, wall time | Throughput | Peak RAM |
|---|---|---|---|
| 1 | 266.8 s | 12.3 chars/s | 9.2 GB |
| 3 | 115.3 s | 28.5 chars/s | 9.2 GB |
| 6 | 171.0 s | 38.5 chars/s | 9.2 GB |

Note the flat memory: every worker shares a single model instance, so raising
`OCR_WORKERS` costs almost no RAM. Scaling flattens past ~4 because the work
becomes memory-bandwidth bound. Start at about a third of your core count.

Lowering `OCR_MAX_DIM` from 1536 to 1024 is a further **1.7x** (94.9 s vs
172.7 s for the same page) at a small cost in detail — worth it if you want
speed, keep 1536 if you want every faint mark.

The two combine: a five-page paper that took ~13 minutes at the stock settings
now takes roughly 2–3 minutes.

`tools/bench.py` and `tools/concurrency.py` reproduce these numbers on your own
machine:

```bash
.venv/bin/python tools/concurrency.py --image samples/crop.jpg --jobs 4 --mode shared
```

### Why not the GPU?

The machine has an **Intel Arc 140T integrated GPU** and no NVIDIA GPU, running
under WSL2. That is a hard combination:

* WSL2 exposes `/dev/dxg` (a D3D12 paravirtualisation path), which is how CUDA
  reaches a GPU from WSL. There is no `/dev/dri` render node, so the Linux
  Intel compute stack (Level Zero / oneAPI / OpenVINO GPU) cannot see the
  iGPU at all.
* PaddlePaddle has no Intel GPU wheel. Its "XPU" support means Baidu's Kunlun
  chips, not Intel Arc.
* An Intel Arc iGPU also shares system RAM with the CPU, so it offers little
  extra memory bandwidth — the actual bottleneck here.

It is not impossible: `PaddleOCRVL` accepts `vl_rec_backend="vllm-server"` plus
`vl_rec_server_url`, so the VLM could be served by an Intel-XPU build of vLLM
in Docker while layout detection stays on CPU. That is a substantial project
(install Docker, pull a multi-GB XPU image, wire up the server backend) with an
uncertain payoff on an integrated GPU. Ask if you want to attempt it.

### Uploads

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` | `25` | Reject photos larger than this. |
| `MAX_PAGES_PER_UPLOAD` | `50` | Files per request; the browser batches larger selections automatically. |

---

## How it works

A single process — no second service to keep alive.

```
app/
  main.py            FastAPI routes + Jinja templates
  worker.py          FIFO queue and OCR worker threads
  ocr.py             engine dispatch: PaddleOCR-VL, or the hybrid
  layout.py          PP-DocLayoutV3 wrapper — the figure boxes, ~2.5 s locally
  markdown_utils.py  normalisation, figure refs, question labelling, merging
  image_host.py      imgbb / local / base64 hosting with a per-task URL cache
  deepseek.py        optional cleanup pass (handwriting removal), figure-safe
  store.py           SQLite: tasks, pages, hosted-image URLs, cleaned markdown
templates/           index (upload), task (progress), edit (figure → question)
static/              app.css, app.js, vendored marked + KaTeX (works offline)
tools/               bench.py, concurrency.py, compare_polish.py
data/                runtime: paper2md.db, uploads/<task>/{pages,images}
```

Figure placement is engine-independent: whichever engine read the page, the
question numbers it produced go into the mapping and the renderer does the rest.

**Flow.** Uploading a page writes a JPEG to `data/uploads/<task>/pages/` and
queues a job. A worker thread resizes the photo (and honours EXIF rotation),
runs PaddleOCR-VL, copies the figures it found into `images/`, and stores the
page's markdown. Task status is derived from its pages, so an interrupted run
can be resumed: pages left mid-flight are re-queued on the next start.

**Figure filenames.** Figures are prefixed with their page number
(`p2_img_in_image_box_...jpg`) so two pages that both contain a figure called
`img_in_image_box_0_0_100_100.jpg` cannot overwrite each other. The filename is
the stable key for the figure → question mapping.

**Labelling.** Question numbers are stored as a per-page mapping
(`{filename: "13"}`), never written into the page text — the label is applied
when markdown is rendered. That keeps the format changeable, lets the review
screen override an engine's guess, and means a figure can be re-labelled without
re-running OCR. Setting a figure to `13` moves that figure to just underneath
question 13, emitting `![Q13 附图](url)`. Figures you have not mapped yet stay
exactly where OCR put them — a partial mapping never shuffles the document.

**Download.** Markdown is rebuilt from each page's stored markdown plus its
mapping, pages are wrapped in `<!-- page N -->` markers, and figures are
rewritten to the configured host.

---

## Tests

```bash
.venv/bin/python tests/run.py        # no pytest needed
.venv/bin/pytest tests -q            # if you have pytest
```

`tests/test_flow.py` runs the real app, database, worker thread and markdown
pipeline end-to-end with only the PaddleOCR-VL call stubbed — so it passes even
before Paddle is installed.

---

## Troubleshooting

**"No such image" / figures missing in the preview** — the figure referenced by
the markdown was not on disk. Check the server log at startup; the OCR engine
logs `referenced figure not found on disk: <name>`. Re-run that page with
**Retry**.

**Import error mentioning `libGL.so.1`** — PaddleOCR pulls in OpenCV, which
wants the GL runtime even headless. Install it:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

**A page failed** — click **Retry** on that page's card; other pages are
unaffected. Set `OCR_DEBUG=1` and check the server log for the traceback.

**Very slow** — CPU inference is genuinely slow. Try a lower `OCR_MAX_DIM`
(e.g. `1200`), and make sure nothing else is competing for the CPU.

**Out of memory with `OCR_WORKERS>1`** — each worker loads its own copy of the
model. Go back to `1`.

---

## Notes on the reference project

This app replaces `math-exam-parser` (at
`D:\Family\app\questionExtract\app\math-exam-parser`), reusing its proven parts
and fixing several things along the way:

* **One service instead of two.** The reference split OCR and the web UI into
  two FastAPI processes with polling and a 5-minute result TTL between them.
  Here the queue lives in the web process, so there is one thing to start and
  no window in which a slow page's result can expire before it is collected.
* **Figure filename collisions.** The reference stored every page's figures in
  one shared folder, so identically-named figures from different pages
  overwrote each other. Figures are now prefixed with their page number.
* **Unmapped figures are kept in place.** The reference moved *every* figure to
  the end of the document until it was mapped. Now only mapped figures move.
* **ImgBB instead of Imgur.** The reference's CLI used Imgur, whose anonymous
  uploads are heavily rate-limited (it returns HTTP 429 today); its web UI had
  already moved to ImgBB, which is what this app uses.
* **EXIF rotation** is applied to phone photos, and images are normalised
  before reaching the model.
* **Vendored preview libraries** (marked + KaTeX) so the review screen renders
  formulas without needing a CDN at runtime.
