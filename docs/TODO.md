TODO plan

## 1. Add docstrings
 - module-level docstring
 - functions: mask2image, hex_to_rgb, add_noise, image_rgb2lab, build_bpt_from_image, plot_owen_values
 - classes: BaseSegment, AxisAlignedSegment, BPT, BPT_Segment, Coalition, Explainer
## 2. Fix deprecated NumPy
 - replace np.bool with bool or np.bool_ in Explainer.__init__ and empty_mask.
## 3. Improve exceptions
 - replace generic Exception() with NotImplementedError
 - replace input errors with ValueError / TypeError
## 4. Add type hints
 - especially for image arrays, masks, model function, and return values

## 5. Separate files
- segments.py
- tree.py
- explainer.py
- visualization.py
- utils.py

## 6. Add tests

 - test RGB/grayscale BPT creation
 - test invalid image dtype
 - test output shape of explain_instance
 - test AA vs BPT mode
## 7. Add examples
 - simple PyTorch image classifier example
 - ShapBPT vs axis-aligned baseline
 - visualization notebook