ShapBPT documentation
=====================

**ShapBPT** is a Python package for generating **data-aware image feature attributions**
using Owen values over **Binary Partition Trees (BPTs)**.

Unlike axis-aligned hierarchical partitioning, ShapBPT constructs a hierarchy
directly from the image morphology, enabling explanations that better align
with meaningful visual structures.

.. note::

   ShapBPT is model-agnostic and operates through a masking function interface,
   making it applicable to a wide range of black-box models.

---

🚀 Key Features
---------------

* Data-aware hierarchical explanation using Binary Partition Trees (BPT)
* Owen-value based recursive attribution
* Model-agnostic design via masking functions
* Axis-aligned baseline (AA) for comparison
* Supports real-world pipelines (ImageNet, detection, anomaly detection)

---

⚡ Quick example
---------------

.. code-block:: python

   import shap_bpt

   explainer = shap_bpt.Explainer(fm, image, num_explained_classes=4)

   shap_values = explainer.explain_instance(
       max_evals=1000,
       method="BPT"
   )

   shap_bpt.plot_owen_values(explainer, shap_values, class_names)

---

📦 Installation
---------------

.. code-block:: bash

   pip install shap-bpt

---

📚 Documentation Overview
------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   masking_function

.. toctree::
   :maxdepth: 2
   :caption: User guide

   theory
   explainer
   plotting

.. toctree::
   :maxdepth: 2
   :caption: Examples

   notebooks/index

.. toctree::
   :maxdepth: 2
   :caption: API examples

   api_examples/imagenet_bpt
   api_examples/imagenet_aa

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Development

   repo_map