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
exclude_patterns = []

nb_execution_mode = "off"

autosummary_generate = True

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

