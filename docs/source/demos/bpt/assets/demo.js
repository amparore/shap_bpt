// The ShapBPT demo (demos/bpt/index.rst): how a Binary Partition Tree is built from a photo, and
// how ShapBPT splits it, top down, to explain what a model predicts.
//
// Everything is precomputed by ../compute (explain.py logs every split of the library's own
// explainer, pack.py writes data.js); the page only replays it:
//
// - "Build the tree": the BPT's merges, bottom up, from K small regions to the whole photo, with
//   the top of the tree as an icicle plot. Point at a region; scroll (or - and +) to climb the
//   tree from it.
// - "Explain with it": the explainer's splits, top down, as it spends its budget of model calls.
//   For the last split it shows the four inputs of the Owen formula, S, S + A, S + B and S + A + B
//   (absent pixels are blurred, as in the masking function), and the attribution map so far.
// - "BPT vs grid": the same budget spent on the BPT and on axis-aligned splits (AA).
//
// Without JavaScript, the <noscript> figure shows; when the data does not load, the script shows it.
(() => {
  const root = document.querySelector(".shapbpt-demo");
  if (!root) return;
  const fallback = root.querySelector("noscript")?.textContent ?? "";
  const fail = (error) => {
    if (error) console.error("The ShapBPT demo could not start:", error);
    root.classList.add("shapbpt-demo-failed");
    root.innerHTML = fallback;
  };

  const S = 224; // the photos' size
  const Z = 2; // canvas pixels per photo pixel
  const W = S * Z;
  const SVG = "http://www.w3.org/2000/svg";
  const POS = [214, 39, 40]; // raises the logit
  const NEG = [31, 119, 180]; // lowers it
  const ICICLE_DEPTH = 9;

  const begin = () => {
    try {
      start(window.SHAPBPT_DEMO);
    } catch (error) {
      fail(error);
    }
  };
  if (window.SHAPBPT_DEMO) queueMicrotask(begin);
  else {
    const loader = document.createElement("script");
    loader.src = `${root.dataset.root ?? ""}data.js?v=${root.dataset.version ?? ""}`;
    loader.onload = begin;
    loader.onerror = () => fail();
    document.head.appendChild(loader);
  }

  // ---------------------------------------------------------------------------------------------
  // small helpers

  function el(tag, cls, ...children) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    for (const c of children) if (c != null) node.append(c);
    return node;
  }
  function bytes(s) {
    const b = atob(s);
    const a = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i += 1) a[i] = b.charCodeAt(i);
    return a.buffer;
  }
  const u16 = (s) => new Uint16Array(bytes(s));
  const u32 = (s) => new Uint32Array(bytes(s));
  const i32 = (s) => new Int32Array(bytes(s));
  const fmt = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
  const int = (x) => Math.round(x).toLocaleString("en-US");
  const pct = (x) => `${(100 * x).toFixed(x < 0.01 ? 2 : 1)}%`;
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }
  function pixelsOf(img) {
    const c = document.createElement("canvas");
    c.width = S;
    c.height = S;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, S, S);
    return ctx.getImageData(0, 0, S, S).data;
  }
  function canvas(cls, label) {
    const c = el("canvas", cls);
    c.width = W;
    c.height = W;
    c.setAttribute("role", "img");
    if (label) c.setAttribute("aria-label", label);
    return c;
  }
  function pointOf(event, node) {
    const r = node.getBoundingClientRect();
    const x = Math.floor(((event.clientX - r.left) / r.width) * S);
    const y = Math.floor(((event.clientY - r.top) / r.height) * S);
    return x >= 0 && y >= 0 && x < S && y < S ? y * S + x : -1;
  }

  // ---------------------------------------------------------------------------------------------
  // a photo's tree and logs, decoded once

  // a photo's tree and logs are in data.js (the first) or in their own file, loaded when chosen
  function loadPhoto(p) {
    const loaded = () => window.SHAPBPT_DEMO_PHOTOS?.[p.id];
    if (p.base || loaded()) return Promise.resolve(Object.assign(p, loaded()));
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `${root.dataset.root ?? ""}${p.file}?v=${root.dataset.version ?? ""}`;
      script.onload = () => (loaded() ? resolve(Object.assign(p, loaded())) : reject(new Error(p.file)));
      script.onerror = () => reject(new Error(`could not load ${p.file}`));
      document.head.appendChild(script);
    });
  }

  async function prepare(p) {
    if (p._ready) return p._ready;
    p._ready = (async () => {
      await loadPhoto(p);
      const [photo, blur] = await Promise.all([loadImage(p.photo), loadImage(p.blur)]);
      const rgb = pixelsOf(photo);
      const blurred = pixelsOf(blur);
      const K = p.K;
      const M = 2 * K - 1; // nodes of the kept tree: K base regions, K - 1 merges
      const base = new Int32Array(S * S);
      const runs = u16(p.base);
      for (let i = 0, at = 0; i < runs.length; i += 2) {
        base.fill(runs[i], at, at + runs[i + 1]);
        at += runs[i + 1];
      }
      const merges = u16(p.merges);
      const parent = new Int32Array(M).fill(-1);
      const left = new Int32Array(M).fill(-1);
      const right = new Int32Array(M).fill(-1);
      for (let m = 0; m < K - 1; m += 1) {
        const n = K + m;
        left[n] = merges[2 * m];
        right[n] = merges[2 * m + 1];
        parent[left[n]] = n;
        parent[right[n]] = n;
      }
      const area = new Float64Array(M);
      const sum = new Float64Array(3 * M);
      for (let i = 0; i < S * S; i += 1) {
        const b = base[i];
        area[b] += 1;
        sum[3 * b] += rgb[4 * i];
        sum[3 * b + 1] += rgb[4 * i + 1];
        sum[3 * b + 2] += rgb[4 * i + 2];
      }
      for (let n = K; n < M; n += 1) {
        area[n] = area[left[n]] + area[right[n]];
        for (let c = 0; c < 3; c += 1) sum[3 * n + c] = sum[3 * left[n] + c] + sum[3 * right[n] + c];
      }
      const color = new Uint8ClampedArray(3 * M);
      for (let n = 0; n < M; n += 1) for (let c = 0; c < 3; c += 1) color[3 * n + c] = sum[3 * n + c] / area[n];
      const depth = new Int32Array(M);
      for (let n = M - 2; n >= 0; n -= 1) depth[n] = depth[parent[n]] + 1;
      // the icicle's layout: every node spans a share of its parent, left child first
      const x0 = new Float64Array(M);
      for (let n = M - 1; n >= K; n -= 1) {
        x0[left[n]] = x0[n];
        x0[right[n]] = x0[n] + area[left[n]] / (S * S);
      }
      const tree = { K, M, base, parent, left, right, area, color, depth, x0, rgb, blurred };
      for (const method of ["BPT", "AA"]) {
        const log = p[method];
        const f = Float64Array.from(new Float32Array(bytes(log.f)));
        const coalitions = u32(log.coalitions);
        const d = u16(log.depth);
        const contrib = new Float64Array(coalitions.length);
        for (let c = 0; c < coalitions.length; c += 1) contrib[c] = (f[2 * c] - f[2 * c + 1]) * 0.5 ** d[c];
        tree[method] = {
          shapes: u16(log.shapes),
          parents: i32(log.parents),
          coalitions,
          depth: d,
          f,
          contrib,
          events: u32(log.events),
        };
      }
      // a common scale for the attribution maps: the strongest pixels of either final map
      tree.scale = Math.max(...["BPT", "AA"].map((m) => quantile(heat(tree, m, Infinity).values, 0.995)));
      return tree;
    })();
    return p._ready;
  }

  function quantile(values, q) {
    const a = Float64Array.from(values, Math.abs).sort();
    return a[Math.floor(q * (a.length - 1))] || 1;
  }

  // the coalitions open after `steps` splits, and the attribution map and finest segments they make
  function openAfter(tree, method, steps) {
    const log = tree[method];
    const n = Math.min(steps, log.events.length / 2);
    const open = new Uint8Array(log.coalitions.length);
    open[0] = 1;
    for (let e = 0; e < n; e += 1) {
      open[log.events[2 * e]] = 0;
      open.fill(1, log.events[2 * e + 1], log.events[2 * e + 1] + 4);
    }
    return { open, n };
  }

  function heat(tree, method, steps) {
    const log = tree[method];
    const { open, n } = openAfter(tree, method, steps);
    const values = new Float64Array(S * S);
    const finest = new Int32Array(S * S).fill(-1);
    if (method === "BPT") {
      const acc = new Float64Array(tree.M);
      const isOpen = new Uint8Array(tree.M);
      for (let c = 0; c < open.length; c += 1) {
        if (!open[c]) continue;
        const node = log.shapes[log.coalitions[c]];
        acc[node] += log.contrib[c] / tree.area[node];
        isOpen[node] = 1;
      }
      const label = new Int32Array(tree.M).fill(-1);
      for (let k = tree.M - 1; k >= 0; k -= 1) {
        const p = tree.parent[k];
        if (p >= 0) {
          acc[k] += acc[p];
          label[k] = isOpen[k] ? k : label[p];
        } else label[k] = isOpen[k] ? k : -1;
      }
      for (let i = 0; i < S * S; i += 1) {
        values[i] = acc[tree.base[i]];
        finest[i] = label[tree.base[i]];
      }
    } else {
      const diff = new Float64Array((S + 1) * (S + 1));
      const rects = [];
      for (let c = 0; c < open.length; c += 1) {
        if (!open[c]) continue;
        const s = log.coalitions[c];
        const [y0, y1, x0, x1] = log.shapes.subarray(4 * s, 4 * s + 4);
        const v = log.contrib[c] / ((y1 - y0) * (x1 - x0));
        diff[y0 * (S + 1) + x0] += v;
        diff[y0 * (S + 1) + x1] -= v;
        diff[y1 * (S + 1) + x0] -= v;
        diff[y1 * (S + 1) + x1] += v;
        rects.push([y0, y1, x0, x1, (y1 - y0) * (x1 - x0)]);
      }
      for (let y = 0; y <= S; y += 1)
        for (let x = 0; x <= S; x += 1) {
          const k = y * (S + 1) + x;
          if (x > 0) diff[k] += diff[k - 1];
          if (y > 0) diff[k] += diff[k - (S + 1)];
          if (x > 0 && y > 0) diff[k] -= diff[k - (S + 1) - 1];
        }
      for (let y = 0; y < S; y += 1) for (let x = 0; x < S; x += 1) values[y * S + x] = diff[y * (S + 1) + x];
      rects.sort((a, b) => b[4] - a[4]); // the finest last, on top
      rects.forEach(([y0, y1, x0, x1], k) => {
        for (let y = y0; y < y1; y += 1) finest.fill(k, y * S + x0, y * S + x1);
      });
    }
    return { values, finest, n };
  }

  // the pixels of a segment and of the segments present with it (its chain of parents)
  function chainMask(tree, method, segment, out = new Uint8Array(S * S)) {
    const log = tree[method];
    if (method === "BPT") {
      const marked = new Uint8Array(tree.M);
      for (let s = segment; s >= 0; s = log.parents[s]) marked[log.shapes[s]] = 1;
      const inside = new Uint8Array(tree.M);
      for (let k = tree.M - 1; k >= 0; k -= 1) {
        const p = tree.parent[k];
        inside[k] = marked[k] || (p >= 0 && inside[p]) ? 1 : 0;
      }
      for (let i = 0; i < S * S; i += 1) if (inside[tree.base[i]]) out[i] = 1;
    } else {
      for (let s = segment; s >= 0; s = log.parents[s]) {
        const [y0, y1, x0, x1] = log.shapes.subarray(4 * s, 4 * s + 4);
        for (let y = y0; y < y1; y += 1) out.fill(1, y * S + x0, y * S + x1);
      }
    }
    return out;
  }
  function shapeMask(tree, method, segment) {
    const log = tree[method];
    const out = new Uint8Array(S * S);
    if (method === "BPT") {
      const node = log.shapes[segment];
      const inside = new Uint8Array(tree.M);
      inside[node] = 1;
      for (let k = node - 1; k >= 0; k -= 1) if (tree.parent[k] >= 0 && inside[tree.parent[k]]) inside[k] = 1;
      for (let i = 0; i < S * S; i += 1) if (inside[tree.base[i]]) out[i] = 1;
    } else {
      const [y0, y1, x0, x1] = log.shapes.subarray(4 * segment, 4 * segment + 4);
      for (let y = y0; y < y1; y += 1) out.fill(1, y * S + x0, y * S + x1);
    }
    return out;
  }

  // ---------------------------------------------------------------------------------------------
  // painting

  // paint a 2x canvas from a per-pixel color function, with 1 px edges where `labels` change
  const HI = [255, 214, 0];
  function paint(ctx, rgbaAt, labels, edge, hi) {
    const img = ctx.createImageData(W, W);
    const d = img.data;
    for (let Y = 0; Y < W; Y += 1) {
      const y = Y >> 1;
      for (let X = 0; X < W; X += 1) {
        const x = X >> 1;
        const i = y * S + x;
        const o = 4 * (Y * W + X);
        let border = false;
        if (labels && edge) {
          if (X & 1 && x + 1 < S && labels[i] !== labels[i + 1]) border = true;
          else if (Y & 1 && y + 1 < S && labels[i] !== labels[i + S]) border = true;
        }
        if (hi) {
          const inside = hi[i];
          const near =
            (x > 0 && hi[i - 1] !== inside) ||
            (x + 1 < S && hi[i + 1] !== inside) ||
            (y > 0 && hi[i - S] !== inside) ||
            (y + 1 < S && hi[i + S] !== inside);
          if (near && inside) {
            d[o] = HI[0];
            d[o + 1] = HI[1];
            d[o + 2] = HI[2];
            d[o + 3] = 255;
            continue;
          }
        }
        if (border) {
          d[o] = edge[0];
          d[o + 1] = edge[1];
          d[o + 2] = edge[2];
          d[o + 3] = edge[3];
        } else rgbaAt(i, d, o);
        if (hi && !hi[i]) {
          d[o] = d[o] * 0.82 + 18;
          d[o + 1] = d[o + 1] * 0.82 + 18;
          d[o + 2] = d[o + 2] * 0.82 + 18;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function heatColor(v, scale, d, o, under, i) {
    // the photo, greyed, under the attribution color
    const g = 0.3 * under[4 * i] + 0.59 * under[4 * i + 1] + 0.11 * under[4 * i + 2];
    const grey = 205 + 0.32 * (g - 128);
    const a = Math.min(1, Math.abs(v) / scale) ** 0.6 * 0.92;
    const c = v >= 0 ? POS : NEG;
    d[o] = grey * (1 - a) + c[0] * a;
    d[o + 1] = grey * (1 - a) + c[1] * a;
    d[o + 2] = grey * (1 - a) + c[2] * a;
    d[o + 3] = 255;
  }

  // ---------------------------------------------------------------------------------------------

  function start(data) {
    if (!data?.photos?.length) throw new Error("no photos in the data");
    root.innerHTML = "";
    root.classList.add("shapbpt-demo-ready");
    const state = { photo: 0, view: "build", regions: 0, steps: 0, hover: -1, climb: 0, playing: null };

    // header: the photos and the model's prediction
    const head = el("div", "shapbpt-demo-head");
    const photos = el("div", "shapbpt-demo-photos");
    photos.setAttribute("role", "group");
    photos.setAttribute("aria-label", "Photo");
    const photoButtons = data.photos.map((p, k) => {
      const b = el("button", "shapbpt-demo-photo");
      b.type = "button";
      b.title = p.title;
      b.setAttribute("aria-label", p.title);
      const img = el("img");
      img.src = p.photo;
      img.alt = "";
      b.append(img);
      b.addEventListener("click", () => choose(k));
      photos.append(b);
      return b;
    });
    const prediction = el("p", "shapbpt-demo-prediction");
    head.append(photos, prediction);

    // the views
    const views = [
      ["build", "1 · Build the tree"],
      ["explain", "2 · Explain with it"],
      ["compare", "3 · BPT vs grid"],
    ];
    const tabs = el("div", "shapbpt-demo-tabs");
    tabs.setAttribute("role", "tablist");
    const tabButtons = views.map(([id, label]) => {
      const b = el("button", "shapbpt-demo-tab", label);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.addEventListener("click", () => setView(id));
      tabs.append(b);
      return b;
    });

    const body = el("div", "shapbpt-demo-body");
    const live = el("p", "shapbpt-demo-sr");
    live.setAttribute("aria-live", "polite");
    root.append(head, tabs, body, live);

    let tree = null;
    let panel = null; // the current view's handle: { render(), stop? }

    async function choose(k) {
      state.photo = k;
      photoButtons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === k)));
      root.classList.add("shapbpt-demo-loading");
      const p = data.photos[k];
      try {
        tree = await prepare(p);
      } catch (error) {
        console.error(error);
        p._ready = null; // try again on the next click
        root.classList.remove("shapbpt-demo-loading");
        prediction.textContent = "This photo could not be loaded.";
        return;
      }
      root.classList.remove("shapbpt-demo-loading");
      if (state.photo !== k) return; // another photo was chosen meanwhile
      prediction.replaceChildren(
        `${data.model} predicts `,
        el("strong", "", p.label),
        `: logit ${p.full.toFixed(2)} for the photo, ${p.empty.toFixed(2)} when it is all blurred.`,
      );
      state.regions = Math.min(state.regions || 64, tree.K);
      state.steps = Math.min(state.steps || 60, tree.BPT.events.length);
      setView(state.view);
    }

    function setView(id) {
      stop();
      state.view = id;
      state.hover = -1;
      state.climb = 0;
      tabButtons.forEach((b, k) => b.setAttribute("aria-selected", String(views[k][0] === id)));
      body.replaceChildren();
      panel = { build: buildView, explain: explainView, compare: compareView }[id]();
    }

    function stop() {
      if (state.playing) cancelAnimationFrame(state.playing);
      state.playing = null;
      root.querySelectorAll(".shapbpt-demo-play").forEach((b) => {
        b.textContent = "▶";
        b.setAttribute("aria-label", "Play");
      });
    }

    function playButton(onTick) {
      const b = el("button", "shapbpt-demo-play", "▶");
      b.type = "button";
      b.setAttribute("aria-label", "Play");
      b.addEventListener("click", () => {
        if (state.playing) return stop();
        b.textContent = "❚❚";
        b.setAttribute("aria-label", "Pause");
        let last = performance.now();
        const tick = (now) => {
          const more = onTick((now - last) / 1000);
          last = now;
          if (more) state.playing = requestAnimationFrame(tick);
          else stop();
        };
        state.playing = requestAnimationFrame(tick);
      });
      return b;
    }

    function slider(label, min, max, value, onInput) {
      const wrap = el("label", "shapbpt-demo-slider");
      const text = el("span", "shapbpt-demo-slider-label", label);
      const input = el("input");
      input.type = "range";
      input.min = String(min);
      input.max = String(max);
      input.step = "1";
      input.value = String(value);
      input.addEventListener("input", () => {
        stop();
        onInput(Number(input.value));
      });
      wrap.append(text, input);
      return { wrap, input };
    }

    // =============================================================================================
    // 1. Build the tree: the merges, bottom up

    function buildView() {
      const t = tree;
      const stage = el("div", "shapbpt-demo-stage");
      const frame = el("div", "shapbpt-demo-frame");
      const view = canvas("shapbpt-demo-canvas", "The photo split into the regions of the tree at this level");
      frame.append(view);
      const ctx = view.getContext("2d");
      const side = el("div", "shapbpt-demo-side");
      const intro = el(
        "p",
        "shapbpt-demo-text",
        "A BPT is built bottom up: every pixel starts as its own region, and the two most similar neighbouring regions merge, again and again, until one region is left: the whole photo. Slide to replay the last merges.",
      );
      const status = el("p", "shapbpt-demo-status");
      const info = el("div", "shapbpt-demo-info");
      const modeBox = el("div", "shapbpt-demo-toggle");
      let mode = "mosaic";
      const modes = [
        ["mosaic", "Mean colours"],
        ["photo", "Photo"],
      ].map(([id, label]) => {
        const b = el("button", "", label);
        b.type = "button";
        b.addEventListener("click", () => {
          mode = id;
          modes.forEach((x, k) => x.setAttribute("aria-pressed", String(["mosaic", "photo"][k] === id)));
          render();
        });
        modeBox.append(b);
        return b;
      });
      modes[0].setAttribute("aria-pressed", "true");
      // regions on a log scale: the slider runs over log2(regions), merging to the right
      const top = Math.log2(t.K);
      const steps = 400;
      const toRegions = (v) => Math.max(1, Math.round(2 ** (top * (1 - v / steps))));
      const toSlider = (r) => Math.round(steps * (1 - Math.log2(r) / top));
      const s = slider("Merge", 0, steps, toSlider(state.regions), (v) => {
        state.regions = toRegions(v);
        render();
      });
      const controls = el("div", "shapbpt-demo-controls");
      const play = playButton((dt) => {
        if (state.regions <= 1) state.regions = t.K;
        const v = Math.min(steps, Number(s.input.value) + dt * 55);
        s.input.value = String(v);
        state.regions = toRegions(v);
        render();
        return state.regions > 1;
      });
      controls.append(play, s.wrap, modeBox);

      // the icicle: the top of the tree, root at the top, every node as wide as its region
      const icicle = document.createElementNS(SVG, "svg");
      icicle.setAttribute("class", "shapbpt-demo-icicle");
      icicle.setAttribute("viewBox", `0 0 100 ${ICICLE_DEPTH * 10}`);
      icicle.setAttribute("preserveAspectRatio", "none");
      icicle.setAttribute("aria-hidden", "true");
      const rects = new Map();
      for (let n = t.M - 1; n >= 0; n -= 1) {
        if (t.depth[n] >= ICICLE_DEPTH) continue;
        const w = (t.area[n] / (S * S)) * 100;
        if (w < 0.6) continue;
        const r = document.createElementNS(SVG, "rect");
        r.setAttribute("x", String(t.x0[n] * 100 + 0.15));
        r.setAttribute("y", String(t.depth[n] * 10 + 0.6));
        r.setAttribute("width", String(Math.max(0.2, w - 0.3)));
        r.setAttribute("height", "8.8");
        r.setAttribute("fill", `rgb(${t.color[3 * n]},${t.color[3 * n + 1]},${t.color[3 * n + 2]})`);
        r.addEventListener("pointerenter", () => {
          shown = n;
          render(true);
        });
        r.addEventListener("pointerleave", () => {
          shown = -1;
          render(true);
        });
        icicle.append(r);
        rects.set(n, r);
      }
      const icicleBox = el(
        "figure",
        "shapbpt-demo-icicle-box",
        icicle,
        el("figcaption", "", "The top of the tree. Faint nodes are not merged yet; point at one to see it."),
      );

      stage.append(frame, side);
      side.append(intro, status, info, icicleBox);
      body.append(stage, controls);

      let labels = new Int32Array(S * S);
      let nodeLabel = new Int32Array(t.M);
      let shown = -1; // a node pointed at in the icicle
      function regionsAt(r) {
        const done = t.K - r; // merges done since the base partition
        const exists = t.K + done;
        for (let n = t.M - 1; n >= 0; n -= 1) {
          const p = t.parent[n];
          if (n >= exists) nodeLabel[n] = -1;
          else nodeLabel[n] = p < 0 || p >= exists ? n : nodeLabel[p];
        }
        for (let i = 0; i < S * S; i += 1) labels[i] = nodeLabel[t.base[i]];
        return { done, exists };
      }
      function climbFrom(i) {
        // the region at pixel i, `state.climb` levels up (or down, towards the pixel) from this level
        let n = labels[i];
        if (state.climb > 0) for (let k = 0; k < state.climb && t.parent[n] >= 0; k += 1) n = t.parent[n];
        else if (state.climb < 0) {
          const path = [];
          for (let m = t.base[i]; m !== n; m = t.parent[m]) path.push(m);
          n = path[Math.max(0, path.length + state.climb)] ?? n;
        }
        return n;
      }
      function nodeMask(n) {
        const inside = new Uint8Array(t.M);
        inside[n] = 1;
        for (let k = n - 1; k >= 0; k -= 1) if (t.parent[k] >= 0 && inside[t.parent[k]]) inside[k] = 1;
        return inside;
      }

      function render(quiet) {
        const { done, exists } = regionsAt(state.regions);
        const next = done < t.K - 1 ? t.K + done : -1; // the next merge's node
        const focus = shown >= 0 ? shown : state.hover >= 0 ? climbFrom(state.hover) : -1;
        const inFocus = focus >= 0 ? nodeMask(focus) : null;
        const inA = next >= 0 && focus < 0 ? nodeMask(t.left[next]) : null;
        const inB = next >= 0 && focus < 0 ? nodeMask(t.right[next]) : null;
        const col = t.color;
        paint(
          ctx,
          (i, d, o) => {
            const b = t.base[i];
            const n = labels[i];
            let r, g, bl;
            if (mode === "mosaic") {
              r = col[3 * n];
              g = col[3 * n + 1];
              bl = col[3 * n + 2];
            } else {
              r = t.rgb[4 * i];
              g = t.rgb[4 * i + 1];
              bl = t.rgb[4 * i + 2];
            }
            if (inFocus && !inFocus[b]) {
              r = r * 0.3 + 40;
              g = g * 0.3 + 40;
              bl = bl * 0.3 + 40;
            } else if (inA && inA[b]) {
              r = r * 0.45 + 255 * 0.55;
              g = g * 0.45 + 160 * 0.55;
              bl *= 0.45;
            } else if (inB && inB[b]) {
              r *= 0.45;
              g = g * 0.45 + 200 * 0.55;
              bl = bl * 0.45 + 255 * 0.55;
            }
            d[o] = r;
            d[o + 1] = g;
            d[o + 2] = bl;
            d[o + 3] = 255;
          },
          labels,
          [255, 255, 255, mode === "mosaic" ? 110 : 200],
        );
        // the icicle: merged nodes solid, the rest faint, the focus and its parents outlined
        const path = new Set();
        for (let n = focus; n >= 0; n = t.parent[n]) path.add(n);
        for (const [n, r] of rects) {
          r.classList.toggle("shapbpt-demo-unmerged", n >= exists);
          r.classList.toggle("shapbpt-demo-onpath", path.has(n));
          r.classList.toggle("shapbpt-demo-next", n === next && focus < 0);
        }
        const mergeNo = p_first() + done;
        status.replaceChildren(
          el("strong", "", `${int(state.regions)} region${state.regions === 1 ? "" : "s"}`),
          ` · merge ${int(mergeNo)} of ${int(t.K - 1 + p_first())}`,
        );
        if (focus >= 0) {
          const formed = focus >= t.K ? `formed by merge ${int(p_first() + focus - t.K + 1)}` : `one of the ${int(t.K)} regions the first ${int(p_first())} merges made`;
          info.replaceChildren(
            el("strong", "", `${int(t.area[focus])} pixels`),
            ` (${pct(t.area[focus] / (S * S))} of the photo), ${formed}. `,
            el(
              "span",
              "shapbpt-demo-hint",
              shown >= 0 ? "" : "Scroll, or press + and −, to climb the tree from here.",
            ),
          );
        } else if (next >= 0) {
          info.replaceChildren(
            "Next, ",
            el("span", "shapbpt-demo-key shapbpt-demo-key-a", `${int(t.area[t.left[next]])} px`),
            " and ",
            el("span", "shapbpt-demo-key shapbpt-demo-key-b", `${int(t.area[t.right[next]])} px`),
            " merge. Point at a region to see it.",
          );
        } else info.replaceChildren("One region: the root of the tree, the whole photo. Every split ShapBPT makes follows this tree, top down.");
        if (!quiet) live.textContent = `${state.regions} regions`;
      }
      const p_first = () => data.photos[state.photo].firstMerge;

      view.addEventListener("pointermove", (e) => {
        const i = pointOf(e, view);
        if (i !== state.hover) {
          state.hover = i;
          render(true);
        }
      });
      view.addEventListener("pointerleave", () => {
        state.hover = -1;
        state.climb = 0;
        render(true);
      });
      view.addEventListener(
        "wheel",
        (e) => {
          if (state.hover < 0) return;
          e.preventDefault();
          state.climb += e.deltaY < 0 ? 1 : -1;
          state.climb = Math.max(-40, Math.min(40, state.climb));
          render(true);
        },
        { passive: false },
      );
      view.tabIndex = 0;
      view.addEventListener("keydown", (e) => {
        if (e.key === "+" || e.key === "=") state.climb += 1;
        else if (e.key === "-" || e.key === "_") state.climb -= 1;
        else return;
        if (state.hover < 0) state.hover = (S / 2) * S + S / 2;
        e.preventDefault();
        render(true);
      });
      render();
      return { render };
    }

    // =============================================================================================
    // 2. Explain with it: the splits, top down

    function explainView() {
      const t = tree;
      const method = "BPT";
      const log = t[method];
      const E = log.events.length / 2;
      const stage = el("div", "shapbpt-demo-stage");
      const frame = el("div", "shapbpt-demo-frame");
      const view = canvas("shapbpt-demo-canvas", "The attribution map so far, over the photo");
      frame.append(view);
      const ctx = view.getContext("2d");
      const side = el("div", "shapbpt-demo-side");
      const intro = el(
        "p",
        "shapbpt-demo-text",
        "ShapBPT walks the same tree top down. It splits a region AB into its two children A and B and asks the model about both, with the regions S already present: four inputs, of which two are new. The region whose split changes the logit most goes first, so the budget goes where the model looks.",
      );
      const status = el("p", "shapbpt-demo-status");
      const quad = el("div", "shapbpt-demo-quad");
      const cells = ["S", "S + A", "S + B", "S + A + B"].map((label) => {
        const c = document.createElement("canvas");
        c.width = S;
        c.height = S;
        const value = el("span", "shapbpt-demo-quad-value");
        const cell = el("figure", "shapbpt-demo-quad-cell", c, el("figcaption", "", el("span", "", label), value));
        quad.append(cell);
        return { c, ctx: c.getContext("2d"), value, cell };
      });
      const split = el("div", "shapbpt-demo-info");
      const legend = el(
        "p",
        "shapbpt-demo-legend",
        el("span", "shapbpt-demo-swatch shapbpt-demo-swatch-pos"),
        " raises the logit ",
        el("span", "shapbpt-demo-swatch shapbpt-demo-swatch-neg"),
        " lowers it · blurred: absent",
      );
      side.append(intro, status, quad, split, legend);
      stage.append(frame, side);
      const s = slider("Model calls", 0, E, state.steps, (v) => {
        state.steps = v;
        render();
      });
      const controls = el("div", "shapbpt-demo-controls");
      const play = playButton((dt) => {
        if (state.steps >= E) state.steps = 0;
        state.acc = (state.acc ?? 0) + dt * (state.steps < 20 ? 3 : state.steps < 80 ? 12 : 60);
        const add = Math.floor(state.acc);
        state.acc -= add;
        state.steps = Math.min(E, state.steps + add);
        s.input.value = String(state.steps);
        render();
        return state.steps < E;
      });
      controls.append(play, s.wrap);
      body.append(stage, controls);

      let current = null;
      function render(quiet) {
        current = heat(t, method, state.steps);
        const { values, finest } = current;
        const hoverSeg = state.hover >= 0 ? finest[state.hover] : -1;
        paint(
          ctx,
          (i, d, o) => heatColor(values[i], t.scale, d, o, t.rgb, i),
          finest,
          [255, 255, 255, 190],
          hoverSeg >= 0 ? finest.map((v) => (v === hoverSeg ? 1 : 0)) : null,
        );
        const evals = 2 + 2 * state.steps;
        status.replaceChildren(
          el("strong", "", `${int(evals)} model calls`),
          ` · ${int(state.steps)} split${state.steps === 1 ? "" : "s"} · sum of attributions ${fmt(t.BPT.f[0] - t.BPT.f[1])}`,
        );
        // the last split's four inputs
        if (state.steps === 0) {
          const full = new Uint8Array(S * S).fill(1);
          drawMasked(cells[0], new Uint8Array(S * S), null, t.BPT.f[1]);
          drawMasked(cells[3], full, null, t.BPT.f[0]);
          cells[1].cell.classList.add("shapbpt-demo-quad-empty");
          cells[2].cell.classList.add("shapbpt-demo-quad-empty");
          split.replaceChildren(
            "Before any split, the whole photo is one region: all its logit gain, ",
            el("strong", "", fmt(t.BPT.f[0] - t.BPT.f[1])),
            ", from blurred to sharp, is spread evenly over it. Press ▶.",
          );
        } else {
          cells[1].cell.classList.remove("shapbpt-demo-quad-empty");
          cells[2].cell.classList.remove("shapbpt-demo-quad-empty");
          const e = state.steps - 1;
          const c = log.events[2 * e];
          const first = log.events[2 * e + 1];
          const f = log.f;
          const segA = log.coalitions[first]; // S + A
          const segB = log.coalitions[first + 1]; // S + B
          const segAB = log.coalitions[c];
          const parentS = log.parents[segAB];
          const mS = parentS >= 0 ? chainMask(t, method, parentS) : new Uint8Array(S * S);
          const mA = chainMask(t, method, segA);
          const mB = chainMask(t, method, segB);
          const mAB = chainMask(t, method, segAB);
          const onlyA = shapeMask(t, method, segA);
          const onlyB = shapeMask(t, method, segB);
          const fS = f[2 * c + 1];
          const fAB = f[2 * c];
          const fA = f[2 * first];
          const fB = f[2 * (first + 1)];
          drawMasked(cells[0], mS, null, fS);
          drawMasked(cells[1], mA, onlyA, fA, "a");
          drawMasked(cells[2], mB, onlyB, fB, "b");
          drawMasked(cells[3], mAB, null, fAB);
          const w = 0.5 ** log.depth[c];
          const toA = (w / 2) * (fA - fS + (fAB - fB));
          const toB = (w / 2) * (fB - fS + (fAB - fA));
          const areaA = onlyA.reduce((a, b) => a + b, 0);
          const areaB = onlyB.reduce((a, b) => a + b, 0);
          split.replaceChildren(
            `Split ${int(state.steps)}, at depth ${log.depth[c]} (weight 1/${2 ** log.depth[c]}): `,
            el("span", "shapbpt-demo-key shapbpt-demo-key-a", `A, ${int(areaA)} px`),
            " gets ",
            el("strong", "", fmt(toA)),
            ", ",
            el("span", "shapbpt-demo-key shapbpt-demo-key-b", `B, ${int(areaB)} px`),
            " gets ",
            el("strong", "", fmt(toB)),
            ` of AB's ${fmt(w * (fAB - fS))}.`,
          );
        }
        if (state.hover >= 0 && hoverSeg >= 0) {
          let sum = 0;
          let n = 0;
          for (let i = 0; i < S * S; i += 1)
            if (finest[i] === hoverSeg) {
              sum += values[i];
              n += 1;
            }
          frame.dataset.tip = `This region: ${int(n)} px, ${fmt(sum)}`;
        } else delete frame.dataset.tip;
        if (!quiet) live.textContent = `${evals} model calls`;
      }
      function drawMasked(cell, present, outline, value, key) {
        const img = cell.ctx.createImageData(S, S);
        const d = img.data;
        const a = key === "a" ? [255, 160, 0] : [0, 200, 255];
        for (let i = 0; i < S * S; i += 1) {
          const src = present[i] ? t.rgb : t.blurred;
          d[4 * i] = src[4 * i];
          d[4 * i + 1] = src[4 * i + 1];
          d[4 * i + 2] = src[4 * i + 2];
          d[4 * i + 3] = 255;
          if (outline && outline[i]) {
            const x = i % S;
            let edge = false;
            for (let r = 1; r <= 2 && !edge; r += 1)
              edge =
                (x >= r && !outline[i - r]) ||
                (x < S - r && !outline[i + r]) ||
                (i >= r * S && !outline[i - r * S]) ||
                (i < S * S - r * S && !outline[i + r * S]);
            const k = edge ? 1 : 0.38;
            for (let c = 0; c < 3; c += 1) d[4 * i + c] = d[4 * i + c] * (1 - k) + a[c] * k;
          }
        }
        cell.ctx.putImageData(img, 0, 0);
        cell.value.textContent = value.toFixed(2);
      }
      view.addEventListener("pointermove", (e) => {
        const i = pointOf(e, view);
        if (i !== state.hover) {
          state.hover = i;
          render(true);
        }
      });
      view.addEventListener("pointerleave", () => {
        state.hover = -1;
        render(true);
      });
      render();
      return { render };
    }

    // =============================================================================================
    // 3. BPT vs grid: the same budget, two hierarchies

    function compareView() {
      const t = tree;
      const E = Math.min(t.BPT.events.length, t.AA.events.length) / 2;
      const pair = el("div", "shapbpt-demo-pair");
      const sides = ["BPT", "AA"].map((method) => {
        const c = canvas("shapbpt-demo-canvas", `${method} attribution map`);
        const value = el("p", "shapbpt-demo-status");
        const fig = el(
          "figure",
          "shapbpt-demo-frame shapbpt-demo-frame-half",
          c,
          el(
            "figcaption",
            "",
            el("strong", "", method === "BPT" ? "BPT" : "Grid (AA)"),
            method === "BPT" ? " · regions from the photo" : " · halves of rectangles",
          ),
          value,
        );
        pair.append(fig);
        return { method, c, ctx: c.getContext("2d"), value };
      });
      const text = el(
        "p",
        "shapbpt-demo-text",
        "The same explainer and the same number of model calls, on two hierarchies: the photo's BPT, and the axis-aligned splits of a grid (method=\"AA\"). Point at either map to compare the region under the pointer.",
      );
      const s = slider("Model calls", 1, E, Math.max(1, Math.min(state.steps, E)), (v) => {
        state.steps = v;
        render();
      });
      const controls = el("div", "shapbpt-demo-controls");
      const play = playButton((dt) => {
        if (state.steps >= E) state.steps = 1;
        state.acc = (state.acc ?? 0) + dt * (state.steps < 80 ? 12 : 60);
        const add = Math.floor(state.acc);
        state.acc -= add;
        state.steps = Math.min(E, state.steps + add);
        s.input.value = String(state.steps);
        render();
        return state.steps < E;
      });
      controls.append(play, s.wrap);
      body.append(text, pair, controls);

      function render(quiet) {
        for (const side of sides) {
          const { values, finest, n } = heat(t, side.method, state.steps);
          const seg = state.hover >= 0 ? finest[state.hover] : -1;
          paint(
            side.ctx,
            (i, d, o) => heatColor(values[i], t.scale, d, o, t.rgb, i),
            finest,
            [255, 255, 255, 190],
            seg >= 0 ? finest.map((v) => (v === seg ? 1 : 0)) : null,
          );
          if (seg >= 0) {
            let sum = 0;
            let k = 0;
            for (let i = 0; i < S * S; i += 1)
              if (finest[i] === seg) {
                sum += values[i];
                k += 1;
              }
            side.value.replaceChildren(`Region under the pointer: ${int(k)} px, `, el("strong", "", fmt(sum)));
          } else {
            let pos = 0;
            for (let i = 0; i < S * S; i += 1) if (values[i] > 0) pos += values[i];
            side.value.replaceChildren(`${int(2 + 2 * n)} model calls · positive mass `, el("strong", "", fmt(pos)));
          }
        }
        if (!quiet) live.textContent = `${2 + 2 * state.steps} model calls`;
      }
      for (const side of sides) {
        side.c.addEventListener("pointermove", (e) => {
          const i = pointOf(e, side.c);
          if (i !== state.hover) {
            state.hover = i;
            render(true);
          }
        });
        side.c.addEventListener("pointerleave", () => {
          state.hover = -1;
          render(true);
        });
      }
      render();
      return { render };
    }

    choose(0).catch(fail);
  }
})();
