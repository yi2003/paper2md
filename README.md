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
5. **Download** — the whole paper as `.md`, or a single page if you prefer.

Figures the model found are copied into the markdown as
`*[Q13 附图]*` immediately followed by the image, so a reader can always tell
which diagram goes with which question.

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
  ocr.py             PaddleOCR-VL wrapper: resize → predict → extract figures
  markdown_utils.py  normalisation, figure refs, question labelling, merging
  image_host.py      imgbb / local / base64 hosting with a per-task URL cache
  store.py           SQLite: tasks, pages, hosted-image URLs
templates/           index (upload), task (progress), edit (figure → question)
static/              app.css, app.js, vendored marked + KaTeX (works offline)
data/                runtime: paper2md.db, uploads/<task>/{pages,images}
```

**Flow.** Uploading a page writes a JPEG to `data/uploads/<task>/pages/` and
queues a job. A worker thread resizes the photo (and honours EXIF rotation),
runs PaddleOCR-VL, copies the figures it found into `images/`, and stores the
page's markdown. Task status is derived from its pages, so an interrupted run
can be resumed: pages left mid-flight are re-queued on the next start.

**Figure filenames.** Figures are prefixed with their page number
(`p2_img_in_image_box_...jpg`) so two pages that both contain a figure called
`img_in_image_box_0_0_100_100.jpg` cannot overwrite each other. The filename is
the stable key for the figure → question mapping.

**Labelling.** Setting a figure to `13` moves that figure to just underneath
question 13 and tags it `*[Q13 附图]*`. Figures you have not mapped yet stay
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
