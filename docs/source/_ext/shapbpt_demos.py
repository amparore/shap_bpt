"""Sphinx extension for the demos (``demos/``): interactive pages computed once with ShapBPT.

Adapted from the demos extension of shapiq (https://github.com/mmschlk/shapiq,
``docs/source/_ext/shapiq_demos.py``, MIT License, Copyright (c) 2023 the shapiq developers).

Every demo has a folder ``demos/<name>/``:

- ``index.rst``: the demo's page;
- ``assets/``: what the page loads, ``demo.js``, ``demo.css`` and ``data.js`` (the precomputed
  data), copied to ``_static/demos/<name>/`` of the built docs;
- ``compute/``: the scripts that computed the data, and how to run them again (not part of the
  docs: see ``exclude_patterns`` in ``conf.py``).

The ``shapbpt-demo`` directive embeds a demo in any page::

    .. shapbpt-demo:: bpt
       :anchor: bpt-demo
       :fallback: docs_files-header_image.png
       :alt: What the fallback image shows.

It writes ``<div class="shapbpt-demo" data-demo="<name>" data-root="..." data-version="...">``,
where ``data-root`` is the URL of the demo's assets relative to the page and ``data-version`` a
checksum of ``data.js``, and adds ``demo.css``, ``data.js`` and ``demo.js`` to the page (the data
deferred just before the script). Without JavaScript, the container shows the fallback image, a
file of ``_static/``. Builders other than HTML leave demos out.
"""

from __future__ import annotations

import shutil
import zlib
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri

if TYPE_CHECKING:
    from sphinx.application import Sphinx

_ASSETS = ("demo.js", "demo.css")
_DATA = "data.js"


class DemoNode(nodes.General, nodes.Element):
    """A demo's container; the demo's script fills it."""


class DemoDirective(SphinxDirective):
    """Embed a demo: ``.. shapbpt-demo:: <name>``, the name of its folder in ``demos/``."""

    required_arguments = 1
    option_spec: ClassVar[dict[str, Any]] = {
        "anchor": directives.unchanged_required,
        "fallback": directives.unchanged_required,
        "alt": directives.unchanged_required,
    }

    def run(self) -> list[nodes.Node]:
        name = self.arguments[0]
        assets = Path(self.env.srcdir) / "demos" / name / "assets"
        if missing := [f for f in _ASSETS if not (assets / f).is_file()]:
            raise self.error(f"the demo {name!r} has no {', '.join(missing)} in demos/{name}/assets")
        for file in (*_ASSETS, _DATA):
            if (assets / file).is_file():
                self.env.note_dependency(str(assets / file))
        data = assets / _DATA
        version = f"{zlib.crc32(data.read_bytes()):08x}" if data.is_file() else ""
        node = DemoNode(demo=name, fallback=self.options.get("fallback"), version=version)
        node["alt"] = self.options.get("alt", "")
        if "anchor" in self.options:
            node["ids"].append(self.options["anchor"])
        return [node]


def _visit_demo(self, node: DemoNode) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    root = relative_uri(page, f"_static/demos/{node['demo']}/")
    self.body.append(
        self.starttag(
            node,
            "div",
            CLASS="shapbpt-demo",
            **{"data-demo": node["demo"], "data-root": root, "data-version": node["version"]},
        )
    )
    if node["fallback"]:
        src = relative_uri(page, f"_static/{node['fallback']}")
        self.body.append(
            f'<noscript><img class="shapbpt-demo-fallback" src="{escape(src)}"'
            f' alt="{escape(node["alt"])}"></noscript>'
        )


def _depart_demo(self, node: DemoNode) -> None:  # noqa: ARG001
    self.body.append("</div>\n")


def _skip_demo(self, node: DemoNode) -> None:  # noqa: ARG001
    raise nodes.SkipNode


def _add_assets(app: Sphinx, pagename, templatename, context, doctree) -> None:  # noqa: ARG001
    """Add the stylesheet, data and script of every demo on a page to it."""
    if doctree is None:
        return
    for name in sorted({node["demo"] for node in doctree.findall(DemoNode)}):
        app.add_css_file(f"demos/{name}/demo.css", priority=900)
        if (Path(app.srcdir) / "demos" / name / "assets" / _DATA).is_file():
            app.add_js_file(f"demos/{name}/{_DATA}", loading_method="defer", priority=499)
        app.add_js_file(f"demos/{name}/demo.js", loading_method="defer")


def _copy_assets(app: Sphinx, env) -> None:  # noqa: ARG001
    """Copy every demo's assets to ``_static/demos/<name>/`` before the pages are written."""
    if app.builder.format != "html":
        return
    for assets in sorted((Path(app.srcdir) / "demos").glob("*/assets")):
        target = Path(app.outdir) / "_static" / "demos" / assets.parent.name
        shutil.copytree(assets, target, dirs_exist_ok=True)


def setup(app: Sphinx) -> dict[str, Any]:
    skip = (_skip_demo, None)
    app.add_node(DemoNode, html=(_visit_demo, _depart_demo), latex=skip, text=skip, man=skip, texinfo=skip)
    app.add_directive("shapbpt-demo", DemoDirective)
    app.connect("html-page-context", _add_assets)
    app.connect("env-updated", _copy_assets)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
