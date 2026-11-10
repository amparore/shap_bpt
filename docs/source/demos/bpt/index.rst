Inside ShapBPT: Build the Tree, Then Explain With It
====================================================

How does ShapBPT decide which parts of a photo to explain, and in what order? This demo replays,
step by step, the two halves of the method on three photos and a ResNet-50: how the photo's
**Binary Partition Tree** (BPT) is built, bottom up, and how the explainer walks that tree, top
down, spending its budget of model calls where the prediction changes most.

.. shapbpt-demo:: bpt
   :anchor: bpt-demo
   :fallback: docs_files-header_image.png
   :alt: ShapBPT's explanation of a photo, compared with an axis-aligned explanation.

What you see
------------

**1 · Build the tree.** A BPT starts with every pixel as a region of its own, and merges the two
most similar neighbouring regions, again and again, until a single region is left: the whole
photo. The slider replays the last merges, from a few thousand small regions to one (the first
merges, of single pixels, are too small to see). *Mean colours* paints every region in its average
colour, which is what the merging compares. Point at a region to see it, and scroll (or press
``+`` and ``−``) to climb the tree from it: up to the larger regions it is merged into, down to
the smaller ones it was made of. Beside the photo, the top of the tree is drawn as an icicle plot,
the root at the top and every node as wide as its region; the nodes not merged yet are faint.

**2 · Explain with it.** ShapBPT computes Owen values over the tree. It starts with the whole
photo as one coalition: everything the model gains from the blurred photo to the sharp one,
``f(photo) − f(blurred)``, is spread over it. Then it splits a region ``AB`` into its two children
``A`` and ``B`` and asks the model about them, with the regions ``S`` that are present around them
in this branch of the recursion: the four inputs of the Owen formula, ``S``, ``S + A``, ``S + B``
and ``S + A + B``, of which two are new model calls. ``A`` receives

.. math::

   \frac{w}{2}\Big[\big(f(S \cup A) - f(S)\big) + \big(f(S \cup A \cup B) - f(S \cup B)\big)\Big]

and ``B`` the same with ``A`` and ``B`` swapped, where :math:`w = 2^{-d}` halves at every depth
:math:`d`. The split with the largest change goes first, so the budget follows the regions the
model relies on. The map shows the attributions after every split; they always add up to
``f(photo) − f(blurred)``.

**3 · BPT vs grid.** The same explainer, with the same number of model calls, on the photo's BPT
and on axis-aligned splits (``method="AA"``), which halve rectangles regardless of what they show.
Point at either map to compare the region under the pointer.

How it was computed
-------------------

The model is timm's ResNet-50 (``resnet50.a1_in1k``), and the explained output is the logit of its
top class. The masking function replaces absent pixels with the photo blurred by a Gaussian of
:math:`\sigma = 8` pixels, one of the backgrounds of the ImageNet notebook:

.. code-block:: python

   import numpy as np
   import torch
   from skimage.filters import gaussian

   import shap_bpt

   background = (gaussian(image, sigma=8, channel_axis=-1) * 255).astype(np.uint8)

   @torch.no_grad()
   def fm(masks):  # (N, H, W) booleans: True where the pixel is present
       m = torch.as_tensor(masks, dtype=torch.float32)[:, None]
       x = to_tensor(image) * m + to_tensor(background) * (1 - m)
       return model(normalize(x)).numpy()

   explainer = shap_bpt.Explainer(fm, image, num_explained_classes=1)
   values = explainer.explain_instance(1500, method="BPT", batch_size=32)

The demo is a replay of exactly this call. The compute script wraps
``shap_bpt.shap_bpt.Coalition.prepare_split`` to log every split the unchanged
``explain_instance`` makes, and checks that the log rebuilds the explanation it returns. The page
then only looks values up: what it shows after *n* model calls is the explainer's state after
those calls.

Run it again
------------

The scripts are in ``docs/source/demos/bpt/compute``. ``config.toml`` lists the photos and the
budget; change it to explain other photos, and run:

.. code-block:: bash

   python explain.py --weights resnet50_a1_0-14fe96d1.pth  # the explanations, a few minutes each
   python pack.py                                          # the demo's data, ../assets/data.js
