# Demos

Interactive pages of the docs: what ShapBPT does on a real model and real photos, computed once
and replayed in the browser. A demo is not run when the docs build: its data is computed by hand
(`compute/`) and committed (`assets/`). The structure follows the demos of
[shapiq](https://github.com/mmschlk/shapiq/tree/main/docs/source/demos) (MIT License).

## One folder per demo

```
demos/
├── README.md         this file
└── bpt/
    ├── index.rst     the demo's page: `.. shapbpt-demo:: bpt` and its write-up
    ├── assets/       what the page loads: demo.js, demo.css, data.js, photo-<id>.js
    └── compute/      the scripts that computed the data, see below
```

The `shapbpt-demo` directive (`_ext/shapbpt_demos.py`) embeds a demo in any page, its own or
another one (for example the landing page):

```rst
.. shapbpt-demo:: bpt
   :anchor: bpt-demo
   :fallback: docs_files-header_image.png
   :alt: What the fallback image shows.
```

It writes `<div class="shapbpt-demo" data-root="..." data-version="...">`, copies `assets/` to
`_static/demos/<name>/`, and adds `demo.css`, `data.js` and `demo.js` to the page. `data-version`
is a checksum of `data.js`, so a new data file is never served from a browser's cache. Without
JavaScript the container shows the fallback image (a file of `_static/`).

## The BPT demo

`compute/explain.py` runs the library's own `Explainer.explain_instance` (BPT and AA) on every
photo of `compute/config.toml`. It wraps `Coalition.prepare_split` to log every split and asserts
that the log rebuilds the returned explanation exactly. `compute/pack.py` keeps the last 4,095
merges of each photo's BPT, the base partition they start from and the split logs, and writes
them to `assets/` (the first photo in `data.js`, the others in `photo-<id>.js`, loaded when
chosen).

```bash
cd docs/source/demos/bpt/compute
pip install timm torch scikit-image shap-bpt
# timm's ResNet-50 (resnet50.a1_in1k), from the timm releases on GitHub:
curl -LO https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth
python explain.py --weights resnet50_a1_0-14fe96d1.pth   # a few minutes per photo on a CPU
python pack.py
```

`explain.py` skips the photos already in `results/`. Neither `results/` nor the weights are
committed (`.gitignore`). To add a photo, add a `[[photos]]` entry to `config.toml` and run both
scripts again.
