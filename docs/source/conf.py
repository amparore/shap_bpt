# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ShapBPT'
copyright = '2026, Elvio Amparore, Muhammad Rashid'
author = 'Elvio Amparore, Muhammad Rashid'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = [
    # the demos' assets and the scripts that computed them are not pages
    "demos/*/assets",
    "demos/*/compute",
    "demos/*/README.md",
]

nb_execution_mode = "off"

autosummary_generate = True

# -- Demos (demos/) --------------------------------------------------------------------------------
# Interactive pages computed once with ShapBPT: one folder per demo, with its page, its assets
# (copied to _static/demos/<name>/) and the scripts that computed its data. The shapbpt-demo
# directive of _ext/shapbpt_demos.py embeds a demo in a page; see demos/README.md.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))
extensions.append("shapbpt_demos")

html_theme = "furo"
html_title = "ShapBPT"

html_static_path = ["_static"]

html_favicon = "_static/logo.png"

html_context = {
    "default_mode": "auto",
}

html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
}

