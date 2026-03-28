Repository map
==============

Main package repository
-----------------------

The main package repository `ShapBPT <https://github.com/amparore/shap_bpt>`_ contains:

* ``shap_bpt/``: the Python package
* ``shap_bpt/shap_bpt.py``: high-level explainer logic and plotting helpers
* ``shap_bpt/bpt.pyx``: Cython implementation used for BPT construction
* ``examples/``: compact demonstration notebooks
* ``docs/``: images and visual assets used in the repository README

Experiments repository
----------------------

The experiments repository `ShapBPT-tests <https://github.com/rashidrao-pk/shap_bpt_tests>`_  contains:

* ``minimal_notebooks/``: small practical usage notebooks
* ``notebooks/``: full experiment pipelines
* experiment folders spanning ImageNet, MS-COCO, CelebA, anomaly detection,
  and human evaluation
* saved results, figures, and supplementary assets for paper reproduction