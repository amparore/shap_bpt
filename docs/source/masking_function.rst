Defining the masking function
=============================

ShapBPT is model-agnostic. Instead of requiring access to model internals,
users provide a masking function that evaluates the model on masked versions of
an image.

Expected interface
------------------

Your masking function should accept a batch of binary masks of shape
``(N, H, W)`` and return a batch of model outputs of shape ``(N, M)``, where:

* ``N`` is the number of masks in the batch,
* ``H`` and ``W`` are the image height and width,
* ``M`` is the number of output scores returned by the model.

Example for a classifier
------------------------

.. code-block:: python

   def f_masked(masks):
       batch = []
       for mask in masks:
           masked = image.copy()
           masked[~mask] = 0
           batch.append(masked)
       batch = np.stack(batch)
       return classifier(batch)

Practical recommendations
-------------------------

* Keep the masking rule consistent across all calls.
* Return raw model outputs in a stable format.
* Batch predictions when possible for speed.
* Ensure the image used by the masking function has the same spatial shape as
  the one passed to the explainer.

Common pitfalls
---------------

* Passing floating-point images where BPT construction expects ``uint8``.
* Returning outputs with inconsistent dimensions.
* Applying a masking rule during explanation that differs from the intended
  model input semantics.