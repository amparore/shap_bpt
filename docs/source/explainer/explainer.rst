Explainer interface
The main entry point is :class:`shap_bpt.Explainer`.

Creating an explainer
---------------------

.. code-block:: python

   explainer = shap_bpt.Explainer(
       fm=f_masked,
       image_to_explain=image,
       num_explained_classes=4,
       balance_area=False,
       verbose=False,
   )

Parameters
----------

``fm``
   Black-box masking function.

``image_to_explain``
   Image to explain. In practice, BPT construction expects ``uint8`` image data.

``num_explained_classes``
   Number of top predicted outputs to explain.

``balance_area``
   Whether to include an area-based priority adjustment during recursive
   refinement.

``verbose``
   Enables progress reporting.

Computing explanations
----------------------

.. code-block:: python

   shap_values = explainer.explain_instance(
       max_evals=1000,
       method="BPT",
       bpt=None,
       batch_size=64,
       verbose_plot=False,
       pbar=None,
       min_area=1,
       max_weight=None,
   )

Selected arguments
------------------

``max_evals``
   Budget controlling how many masked evaluations are used.

``method``
   Either ``"BPT"`` or ``"AA"``.

``bpt``
   Optional precomputed BPT object.

``batch_size``
   Number of masks evaluated together.

``min_area``
   Stop splitting regions smaller than this area.

``max_weight``
   Optional early stopping criterion based on coalition weight.