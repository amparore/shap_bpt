Plotting explanations
=====================

ShapBPT provides a built-in visualization helper:

.. code-block:: python

   shap_bpt.plot_owen_values(
       explainer,
       shap_values,
       class_names,
       names=None,
   )

Interpretation
--------------

The plotting utility overlays attribution maps for the explained classes and
uses the original image for context.

Typical workflow
----------------

.. code-block:: python

   shap_values = explainer.explain_instance(1000, method="BPT")
   shap_bpt.plot_owen_values(explainer, shap_values, class_names)

Tips
----

* Use consistent class names with the model outputs.
* Compare ``BPT`` and ``AA`` results side by side when evaluating the effect of
  the hierarchy.
* Save figures explicitly with Matplotlib when preparing reports or papers.