Quick start
===========

This guide walks through the core workflow of ShapBPT for explaining image predictions.

The typical pipeline is:

1. Prepare the image to explain
2. Define a masking-based prediction function
3. Create a :class:`shap_bpt.Explainer`
4. Compute attributions using ``method="BPT"``
5. Visualize the resulting explanation

---

⚡ Minimal working example
--------------------------

.. code-block:: python

   import numpy as np
   import shap_bpt

   # image_to_explain must be an H x W x C image
   image_to_explain = image.astype(np.uint8)

   def f_masked(masks):
       """
       masks: (N, H, W)
       returns: (N, M) model outputs
       """
       batch_scores = []
       for mask in masks:
           masked_image = image_to_explain.copy()
           masked_image[~mask] = 0
           scores = model_predict(masked_image)  # user-defined model
           batch_scores.append(scores)
       return np.array(batch_scores)

   # Create explainer
   explainer = shap_bpt.Explainer(
       f_masked,
       image_to_explain,
       num_explained_classes=4,
       verbose=True,
   )

   # Compute BPT-based explanations
   shap_values = explainer.explain_instance(
       max_evals=1000,
       method="BPT",
       batch_size=4,
   )

   # Visualize results
   shap_bpt.plot_owen_values(
       explainer,
       shap_values,
       class_names=class_names,
   )

---

🧠 Understanding the masking function
-------------------------------------

ShapBPT is model-agnostic. Instead of accessing model internals,
it evaluates the model on masked versions of the image.

Key requirements:

* Input: binary masks of shape ``(N, H, W)``
* Output: model predictions of shape ``(N, M)``
* Masking rule must be consistent across calls

---

📊 Axis-aligned baseline (AA)
-----------------------------

To compare against a standard hierarchical baseline, use:

.. code-block:: python

   shap_values_aa = explainer.explain_instance(
       max_evals=1000,
       method="AA",
       batch_size=4,
   )

---

🔍 BPT vs AA: when to use what
------------------------------

* ``method="BPT"``
  
  * Data-aware hierarchy built from image structure  
  * Better alignment with objects and regions  
  * Recommended for most use cases  

* ``method="AA"``
  
  * Axis-aligned rectangular partitioning  
  * Useful baseline for comparison  
  * Helps evaluate the benefit of data-aware structure  

---

💡 Practical tips
-----------------

* Ensure input image is ``uint8`` for proper BPT construction
* Increase ``max_evals`` for more precise explanations
* Use batching (``batch_size``) for faster evaluation
* Compare ``BPT`` and ``AA`` to validate explanation quality

---

🚀 Next steps
-------------

* See :doc:`notebooks/index` for full notebook examples
* Explore :doc:`explainer/theory` for the Owen-value formulation
* Check :doc:`api` for detailed API documentation