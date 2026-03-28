Installation
============

Requirements
------------

ShapBPT requires Python together with the following core dependencies:

* Cython
* NumPy
* Matplotlib
* tqdm
* scikit-image

Install from PyPI
-----------------

.. code-block:: bash

   pip install shap-bpt

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/amparore/shap_bpt
   cd shap_bpt
   pip install -U pip setuptools wheel
   pip install .

Because the package includes a Cython extension, building from source may
require a working compiler toolchain.

Verify installation
-------------------

.. code-block:: python

   import shap_bpt
   print(shap_bpt.__version__)