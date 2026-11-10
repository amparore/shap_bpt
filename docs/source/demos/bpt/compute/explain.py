"""Explain the demo's photos with ShapBPT (BPT and AA), logging every split the explainer makes.

The library's own ``Explainer.explain_instance`` runs unchanged; ``Coalition.prepare_split`` is
wrapped so that every split is recorded: which coalition was split, the two masked inputs the
model saw (as chains of segments) and the two model outputs. The demo replays this log, so what it
shows at a budget of b evaluations is exactly the state of the explainer after b evaluations.

Writes ``results/<photo>.json`` (and the photo and its masking background as JPEGs).

Usage::

    python explain.py --weights resnet50_a1_0-14fe96d1.pth
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from skimage.filters import gaussian

import shap_bpt
from shap_bpt import shap_bpt as sb

HERE = Path(__file__).resolve().parent
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Recorder:
    """Records segments, coalitions and splits while ``explain_instance`` runs."""

    def __init__(self) -> None:
        self.segments: list[list] = []  # [shape, parent segment or -1]
        self.coalitions: list[list] = []  # [segment, f(S u AB), f(S), depth]
        self.events: list[list] = []  # [split coalition, first of its 4 children]

    def segment(self, seg) -> int:
        if isinstance(seg, sb.BPT_Segment):
            shape = int(seg.index)
        elif isinstance(seg, sb.AxisAlignedSegment):
            shape = [int(seg.ymin), int(seg.ymax), int(seg.xmin), int(seg.xmax)]
        else:  # the base segment: nothing present
            return -1
        if not hasattr(seg, "_demo_id"):  # (not id(): Python reuses the ids of freed objects)
            parent = self.segment(seg.parent)
            seg._demo_id = len(self.segments)
            self.segments.append([shape, parent])
        return seg._demo_id

    def coalition(self, c, depth: int) -> int:
        if not hasattr(c, "_demo_id"):
            c._demo_id = len(self.coalitions)
            c._demo_depth = depth
            self.coalitions.append(
                [self.segment(c.segment), float(c.f_SuAB[0]), float(c.f_S[0]), depth]
            )
        return c._demo_id

    def install(self) -> None:
        original = sb.Coalition.prepare_split
        recorder = self

        def prepare_split(coalition, explainer):
            depth = getattr(coalition, "_demo_depth", 0)
            me = recorder.coalition(coalition, depth)  # the root is recorded at its first split
            m_a, m_b, completer = original(coalition, explainer)

            def logged(f_a, f_b):
                splits = completer(f_a, f_b)
                first = len(recorder.coalitions)
                for child in splits:
                    recorder.coalition(child, depth + 1)
                recorder.events.append([me, first])
                return splits

            return m_a, m_b, logged

        sb.Coalition.prepare_split = prepare_split
        self._original = original

    def uninstall(self) -> None:
        sb.Coalition.prepare_split = self._original


def load_model(weights: Path) -> torch.nn.Module:
    model = timm.create_model("resnet50", pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def explain(photo: dict, model, budget: int, out: Path, classes: dict) -> None:
    rgb = Image.open(HERE / photo["file"]).convert("RGB").resize((224, 224), Image.LANCZOS)
    image = np.asarray(rgb, dtype=np.uint8)
    background = (gaussian(image, sigma=8, channel_axis=-1) * 255).round().astype(np.uint8)
    img_t = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
    bg_t = torch.tensor(background / 255.0, dtype=torch.float32).permute(2, 0, 1)

    @torch.no_grad()
    def fm(masks: np.ndarray) -> np.ndarray:
        """The masking function: absent pixels show the blurred photo. Returns logits."""
        m = torch.as_tensor(masks, dtype=torch.float32)[:, None]
        x = img_t * m + bg_t * (1 - m)
        return model((x - MEAN) / STD).numpy()

    bpt = sb.build_bpt_from_image(image)
    result = {"id": photo["id"], "size": 224}
    for method in ("BPT", "AA"):
        recorder = Recorder()
        recorder.install()
        explainer = shap_bpt.Explainer(fm, image, num_explained_classes=1)
        try:
            values = explainer.explain_instance(
                budget, method=method, bpt=bpt if method == "BPT" else None, batch_size=32
            )[0]
        finally:
            recorder.uninstall()
        # check: the open coalitions at the end of the log rebuild the library's own result
        open_ = set(range(len(recorder.coalitions))) - {e[0] for e in recorder.events}
        rebuilt = np.zeros((224, 224))
        for k in open_:
            seg, f_suab, f_s, depth = recorder.coalitions[k]
            mask = np.zeros((224, 224), dtype=bool)
            shape = recorder.segments[seg][0]
            if method == "BPT":
                sb.BPT_Segment(bpt, shape, sb.BaseSegment()).fill_mask(mask, ascend_hier=False)
            else:
                mask[shape[0] : shape[1], shape[2] : shape[3]] = True
            rebuilt[mask] += (f_suab - f_s) * 0.5**depth / mask.sum()
        error = float(np.abs(rebuilt - values).max())
        assert error < 1e-6, f"the log does not rebuild the explanation ({error})"
        target = int(explainer.output_indexes[0])
        result[method] = {
            "segments": recorder.segments,
            "coalitions": [[s, round(a, 5), round(b, 5), d] for s, a, b, d in recorder.coalitions],
            "events": recorder.events,
        }
        result["target"] = target
        result["label"] = classes[str(target)][1].replace("_", " ")
        result["full"] = float(explainer.base_f_S[0])
        result["empty"] = float(explainer.base_f_0[0])
        print(
            f"{photo['id']} {method}: {len(recorder.events)} splits, "
            f"{result['label']} logit {result['full']:.2f} (blurred {result['empty']:.2f}), "
            f"sum {values.sum():.3f}, rebuilt to {error:.1e}"
        )
    result["tree"] = {
        "U": int(bpt.U),
        "left": bpt.cl_left.astype(int).tolist(),
        "right": bpt.cl_right.astype(int).tolist(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{photo['id']}.json").write_text(json.dumps(result), encoding="utf-8")
    Image.fromarray(image).save(out / f"{photo['id']}.jpg", quality=90)
    Image.fromarray(background).save(out / f"{photo['id']}-blur.jpg", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", type=Path, required=True, help="timm resnet50 weights")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    classes = json.loads((HERE / config["classes"]).read_text(encoding="utf-8"))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = load_model(args.weights)
    for photo in config["photos"]:
        if (args.results / f"{photo['id']}.json").exists():
            print(f"{photo['id']}: already explained")
            continue
        explain(photo, model, config["budget"], args.results, classes)


if __name__ == "__main__":
    main()
