import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from queue import PriorityQueue
from tqdm.auto import tqdm
import numpy as np
import math, os

######################################################################################################
# Shapley image explanations with data-dependent Binary Partition Trees
######################################################################################################

# Cython implementation of the BPT algorithm
from . import bpt as bpt

######################################################################################################

def mask2image(matrix, color):
    h, w = matrix.shape
    rgb_image = np.zeros((h, w, len(color)))
    # Expand dimensions of color to match the shape of the matrix
    color_expanded = np.expand_dims(np.array(color), axis=(0, 1))
    # Multiply matrix with color along the third dimension
    colored_matrix = matrix[..., np.newaxis] * color_expanded
    # Clip values to ensure they are within [0, 1] range
    colored_matrix = np.clip(colored_matrix, 0, 1)
    return colored_matrix

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255.0 for i in range(0, lv, lv // 3))

######################################################################################################

from matplotlib.colors import LinearSegmentedColormap

# Custom colormap for Shapley values - similar to 'seismic' but with lighter tones.
shapley_values_colormap = LinearSegmentedColormap.from_list("shapley_values_colormap", 
                                                            [(0.0, '#0053d1'),
                                                             (0.2, '#248df4'),
                                                             (0.5, 'white'),  
                                                             (0.8, '#f23754'),
                                                             (1.0, '#cb0021')])

######################################################################################################

class BaseSegment:
    """
    Abstract base class representing a hierarchical image segment.

    Segments define regions of the image that can be recursively
    partitioned and used as feature coalitions during Owen-value
    estimation.
    """
    def __init__(self, parent=None):
        self.parent = parent
        
    def split(self):
        raise Exception()
    
    def fill_mask(self, mat, ascend_hier=True):
        return
        
    def add_inside_coalition(self, shap_values, contrib):
        raise Exception()

    def subtract_outside_coalition(self, shap_values, contrib):
        raise Exception()
        
    def area(self):
        raise Exception()

    def plot(self, ax, color=None):
        raise Exception()

    def contains(self, aa):
        raise Exception()

    def equals(self, aa):
        raise Exception()
    
######################################################################################################
# A symmetric, disjoint, axis-aligned, hierarchical partition 
######################################################################################################

class AxisAlignedSegment(BaseSegment):
    """
    Axis-aligned rectangular image segment.

    This partition strategy recursively splits image regions
    along horizontal or vertical axes and serves as a baseline
    hierarchy independent of image content.
    """
    def __init__(self, xmin, xmax, ymin, ymax, parent):
        super().__init__(parent)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
    #@override
    def split(self, lparent, rparent):
        size_x = self.xmax - self.xmin
        size_y = self.ymax - self.ymin
        assert size_x>=1 or size_y>=1
        lxmin = rxmin = self.xmin
        lxmax = rxmax = self.xmax
        lymin = rymin = self.ymin
        lymax = rymax = self.ymax
        if size_x > size_y and size_x > 1: # split over x
            lxmax = rxmin = (self.xmin + size_x // 2)
        else: # split over y
            lymax = rymin = (self.ymin + size_y // 2)
        lsg = AxisAlignedSegment(lxmin, lxmax, lymin, lymax, lparent)
        rsg = AxisAlignedSegment(rxmin, rxmax, rymin, rymax, rparent)
        # print(f'split {self.area()} -> {lsg.area()} + {rsg.area()}    {self}')
        return (lsg, rsg)
    
    #@override
    def fill_mask(self, mat, ascend_hier=True):
        mat[self.ymin:self.ymax, self.xmin:self.xmax] = True
        if ascend_hier:
            self.parent.fill_mask(mat, ascend_hier) 
        
    #@override
    def add_inside_coalition(self, shap_values, contrib):
        contrib = contrib / self.area()
        for c in range(len(contrib)):
            shap_values[c, self.ymin:self.ymax, self.xmin:self.xmax] += contrib[c]

    #@override
    def subtract_outside_coalition(self, shap_values, contrib):
        if shap_values[0].size==self.area():
            return
        contrib = contrib / (shap_values[0].size - self.area())
        for c in range(len(contrib)):
            shap_values[c, :self.ymin, :] -= contrib[c]
            shap_values[c, self.ymax:, :] -= contrib[c]
            shap_values[c, self.ymin:self.ymax, :self.xmin] -= contrib[c]
            shap_values[c, self.ymin:self.ymax, self.xmax:] -= contrib[c]
        
    #@override
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
    
    #@override
    def plot(self, ax, color=(.3,.7,1.0)):
        ax.add_patch(Rectangle((self.xmin, self.ymin), 
                               self.xmax-self.xmin, self.ymax-self.ymin, 
                               facecolor=color, fill=True, lw=None))

    #@override
    def contains(self, aa):
        return (self.xmin <= aa.xmin and aa.xmax <= self.xmax and
                self.ymin <= aa.ymin and aa.ymax <= self.ymax)

    #@override
    def equals(self, aa):
        return (self.xmin == aa.xmin and aa.xmax == self.xmax and
                self.ymin == aa.ymin and aa.ymax == self.ymax)        

######################################################################################################
# Binary Partition Tree reader (using the code of the AGAT-Team)
######################################################################################################

class BPT:
    """
    Binary Partition Tree representation.

    Stores the hierarchical region structure generated by the
    BinaryPartitionTreeBuilder and provides utilities for
    traversal, serialization, and visualization.
    """
    def __init__(self):
        self.width = self.height = -1
        self.N = self.U = 0
        self.pixels = None
        self.leaf_idx = None
        self.cl_start = self.cl_end = None
        self.cl_left = self.cl_right = None

    def load_from_file(self, bpt_fname):
        with open(bpt_fname, 'r') as f:
            self.width = int(f.readline())
            self.height = int(f.readline())
            self.U = int(f.readline())
            self.N = int(f.readline())
            self.pixels = np.array([int(n) for n in f.readline().split()])
            self.leaf_idx = np.array([int(n) for n in f.readline().split()])
            self.cl_start = np.array([int(n) for n in f.readline().split()])
            self.cl_end = np.array([int(n) for n in f.readline().split()])
            self.cl_left = np.array([int(n) for n in f.readline().split()])
            self.cl_right = np.array([int(n) for n in f.readline().split()])

    def from_bpt_builder(self, bpt_builder):
        enc = bpt_builder.encode()
        (self.width, self.height, self.U, self.N, 
         self.pixels, self.leaf_idx,
         self.cl_start, self.cl_end,
         self.cl_left, self.cl_right) = enc

    def print_tree(self, index=None, lvl=0):
        if index is None: index = self.N-1
        print(' ' * lvl, end='')
        print(f'index={index} ', end='')
        if index < self.U: # leaf node
            pass
            # print(f' pixel {self.pixels[index]}')
        else:
            s = self.cl_start[ index - self.U ]
            e = self.cl_end[ index - self.U ]
            l, r = self.cl_left[ index - self.U ], self.cl_right[ index - self.U ]
            al = 1 if l < self.U else self.cl_end[ l - self.U ] - self.cl_start[ l - self.U ]
            ar = 1 if r < self.U else self.cl_end[ r - self.U ] - self.cl_start[ r - self.U ]
            print(f'  {e-s} -> {al} + {ar}    left={l} right={r}')
            self.print_tree(self.cl_left[ index - self.U ], lvl+1)
            self.print_tree(self.cl_right[ index - self.U ], lvl+1)

######################################################################################################

def add_noise(img, sigma=1.0, alpha=0.5):
    from scipy.ndimage import gaussian_filter
    assert 0.0 <= alpha <= 1.0
    rndgen = np.random.Generator(np.random.PCG64(1234))
    img_noise = rndgen.standard_normal(size=img.shape)*64.0 + 128.0
    img_noise = gaussian_filter(img_noise, sigma=1.0)
    img = np.clip(img*alpha + img_noise*(1.0-alpha), 0.0, 255.0)
    return img

######################################################################################################

def image_rgb2lab(rgb_image):
    from skimage.color import rgb2lab
    lab_image = rgb2lab(rgb_image)# / 255.0)
    # The ranges of Lab values are: L (0:100), a (-128:127), b (-128:127)
    lab_image_scaled = (lab_image + [0, 128, 128]) * (255.0/100.0, 255.0/256.0, 255.0/256.0)
    return lab_image_scaled.astype(np.uint8)

######################################################################################################

# input image is expected to be of type uint8, with shape H*W*3 or H*W*1
def build_bpt_from_image(image, use_lab=True, **kwargs):
    """
    Build a Binary Partition Tree from an input image.

    Parameters
    ----------
    image : np.ndarray
        Input image with dtype `uint8`. Accepted shapes are `(H, W)`,
        `(H, W, 1)`, or `(H, W, 3)`.
    use_lab : bool, default=True
        If True, converts RGB images to Lab color space before building the BPT.
        Lab space often gives more perceptually meaningful image partitions.
    **kwargs
        Additional keyword arguments passed to `BinaryPartitionTreeBuilder`.

    Returns
    -------
    BPT
        A Binary Partition Tree object containing the image hierarchy.

    Raises
    ------
    TypeError
        If the image dtype is not `uint8`.
    ValueError
        If the image shape is not supported.
    """
    if image.dtype!=np.uint8:
        raise Exception('Image pixel type is expected to be uint8.')
    if len(image.shape)==2:
        image = image.reshape((image.shape[0], image.shape[1], 1))
    if len(image.shape)!=3:
        raise Exception('Image shape is expected to be 3-dimensional.')
    if image.shape[2]!=3 and image.shape[2]!=1:
        raise Exception('Image is expected to be RGB (H*W*3) or grayscale (H*W*1).')

    if use_lab:
        image = image_rgb2lab(image)

    bpt_builder = bpt.BinaryPartitionTreeBuilder(image=image, **kwargs)
    bpt_builder.compute()
    bptree = BPT()
    bptree.from_bpt_builder(bpt_builder)
    del bpt_builder
    return bptree

######################################################################################################
# A non-symmetric, disjoint, hierarchical partition of a Binary Partition Tree node
######################################################################################################

class BPT_Segment(BaseSegment):
    """
    Segment corresponding to a node of a Binary Partition Tree.

    Each segment represents a connected image region and can
    be recursively split into its left and right children
    according to the tree structure.
    """
    def __init__(self, bpt, index, parent):
        super().__init__(parent)
        self.bpt = bpt
        self.index = index

    #@override
    def split(self, lparent, rparent):
        if self.area() == 1:
            return None
        ls = BPT_Segment(self.bpt, self.bpt.cl_left[ self.index - self.bpt.U ], lparent)
        rs = BPT_Segment(self.bpt, self.bpt.cl_right[ self.index - self.bpt.U ], rparent)
        return (ls, rs)
    
    #@override
    def fill_mask(self, mat, ascend_hier=True):
        s,e = self.pixels_interval()
        mat.ravel()[ self.bpt.pixels[s:e] ] = True
        if ascend_hier:
            self.parent.fill_mask(mat, ascend_hier)
            
    #@override
    def add_inside_coalition(self, shap_values, contrib):
        contrib = contrib / self.area()
        s,e = self.pixels_interval()
        for c in range(len(contrib)):
            shap_values[c].ravel()[ self.bpt.pixels[s:e] ] += contrib[c]

    #@override
    def subtract_outside_coalition(self, shap_values, contrib):
        if shap_values[0].size==self.area():
            return
        contrib = contrib / (shap_values[0].size - self.area())
        s,e = self.pixels_interval()
        for c in range(len(contrib)):
            shap_values[c].ravel()[ self.bpt.pixels[:s] ] -= contrib[c]
            shap_values[c].ravel()[ self.bpt.pixels[e:] ] -= contrib[c]
         
    #@override
    def plot(self, ax, color=(.3,.7,1.0)):
        img = np.zeros((self.bpt.width, self.bpt.height), dtype=np.int8)
        self.fill_mask(img, ascend_hier=False)
        ax.imshow(mask2image(img, color))

    #@override
    def area(self):
        s,e = self.pixels_interval()
        return float(e - s)

    #@override
    def pixels_interval(self):
        if self.index < self.bpt.U: # leaf node
            return (self.bpt.leaf_idx[self.index], 
                    self.bpt.leaf_idx[self.index] + 1)
        else:
            return (self.bpt.cl_start[ self.index - self.bpt.U ],
                    self.bpt.cl_end[ self.index - self.bpt.U ])

    #@override
    def contains(self, other):
        s1, e1 = self.pixels_interval()
        s2, e2 = other.pixels_interval()
        return s1 <= s2 and e2 <= e1

    #@override
    def equals(self, other):
        s1, e1 = self.pixels_interval()
        s2, e2 = other.pixels_interval()
        return s1 == s2 and e2 == e1

######################################################################################################
# A partition of the features, refined by recursive splitting
# A coalition that is part of the global coalition structure
######################################################################################################

class Coalition:
    """
    Coalition used during recursive Owen-value estimation.

    A coalition stores a segment, associated model predictions,
    and the recursive weight required to compute hierarchical
    Shapley values.
    """
    def __init__(self, explainer, segment, f_SuAB, f_S, weight):
        self.segment = segment  # segment for recursive refinement
        self.f_SuAB = f_SuAB    # contribution with this coalition AB
        self.f_S = f_S          # contribution without this coalition AB
        self.weight = weight    # recursive weight of the Owen formula
        # priority to be split for further partition refinements
        self.priority = -np.max(np.abs(np.subtract(self.f_SuAB, self.f_S))) * self.weight 
        if explainer.balance_area:
            self.priority *= self.segment.area()
    
    def prepare_split(self, explainer):
        # split the current coalition AB into two separate coalitions {A,B}
        coS_A,   coS_B   = self.segment.split(self.segment.parent, self.segment.parent)
        coSuB_A, coSuA_B = self.segment.split(coS_B, coS_A) # flip parents
        assert self.segment.area() == coS_A.area() + coS_B.area()
        
        # build the new masks
        m_SuA, m_SuB = explainer.empty_mask(), explainer.empty_mask()
        coS_A.fill_mask(m_SuA)
        coS_B.fill_mask(m_SuB)
                
        # [f_SuA, f_SuB] = predictions using masks [m_SuA, m_SuB]
        def split_completer(f_SuA, f_SuB):
            # generate the four recursive branches
            phiSuA_S    = Coalition(explainer, coS_A,   f_SuA,       self.f_S, self.weight/2.0)
            phiSuB_S    = Coalition(explainer, coS_B,   f_SuB,       self.f_S, self.weight/2.0)
            phiSuAB_SuA = Coalition(explainer, coSuA_B, self.f_SuAB, f_SuA,    self.weight/2.0)
            phiSuAB_SuB = Coalition(explainer, coSuB_A, self.f_SuAB, f_SuB,    self.weight/2.0)
            splits = [phiSuA_S, phiSuB_S, phiSuAB_SuA, phiSuAB_SuB]
            return splits

        return (m_SuA, m_SuB, split_completer)
        
    def plot(self, ax, explainer, color=(1, 0, 0, 1)):
        m00 = explainer.empty_mask()
        self.segment.parent.fill_mask(m00)
        ax.imshow(mask2image(m00, color))
        ax.axis('off')
        self.segment.plot(ax)
    
    def __lt__(self, other):
        return self.priority < other.priority
    
    def get_shapley(self, shap_values): 
        # compute the weighted marginals and add them to the partition
        contrib = (np.subtract(self.f_SuAB, self.f_S) * self.weight)
        self.segment.add_inside_coalition(shap_values, contrib)

######################################################################################################
# Explainer object. Implementation of the recursive refinement following Owen formula
######################################################################################################

class Explainer:
    """
    Generate ShapBPT explanations for an image and black-box model.

    The explainer estimates Owen/Shapley values over a hierarchical image
    partition. The hierarchy can be either image-adaptive, using a Binary
    Partition Tree, or axis-aligned, using recursive rectangular splits.

    Parameters
    ----------
    fm : callable
        Black-box prediction function. It must accept a batch of boolean masks
        with shape `(N, H, W)` and return prediction scores.
    image_to_explain : np.ndarray
        Input image to explain.
    num_explained_classes : int
        Number of top predicted classes to explain.
    balance_area : bool, default=False
        If True, prioritizes larger/high-impact regions during recursive
        refinement.
    verbose : bool, default=False
        If True, shows a progress bar during explanation.
    """
    def __init__(self, fm, image_to_explain, num_explained_classes, balance_area=False, verbose=False):
        self.fm = fm # black box predictor with masker
        self.image_to_explain = image_to_explain
        self.num_explained_classes = num_explained_classes
        self.balance_area = balance_area
        # foreground prediction (no masking, original input)
        ym = self.fm(np.array([np.ones((self.image_to_explain.shape[0], 
                                        self.image_to_explain.shape[1]), dtype=bool)]))[0]
        self.output_indexes = np.flip(np.argsort(ym))[:self.num_explained_classes]
        self.base_f_S = np.array([float(ym[i]) for i in self.output_indexes])
        # background prediction (everything masked)
        ym = self.fm(np.array([np.zeros((self.image_to_explain.shape[0], 
                                         self.image_to_explain.shape[1]), dtype=bool)]))[0]
        self.base_f_0 = np.array([float(ym[i]) for i in self.output_indexes])
        self.verbose = verbose

    def empty_mask(self, dtype=bool):
        return np.zeros((self.image_to_explain.shape[0], 
                         self.image_to_explain.shape[1]), dtype=dtype)
    
    # get an explanation of the image_to_explain masked by @boolMask
    def predict_masked(self, masks):
        rows = self.fm(np.array(masks))
        f = [[float(ym[i]) for i in self.output_indexes] for ym in rows]
        return np.array(f)
    
    # get the Owen/Shapley coefficients
    def explain_instance(self, max_evals, method='BPT', bpt=None,
                         batch_size=64, verbose_plot=False, pbar=None,
                         min_area=1, max_weight=None):
        """
        Compute Owen/Shapley attribution maps for the image.

        Parameters
        ----------
        max_evals : int
            Maximum number of model evaluations allowed.
        method : {"BPT", "AA"}, default="BPT"
            Partition strategy. `"BPT"` uses image-adaptive Binary Partition Trees;
            `"AA"` uses axis-aligned rectangular splits.
        bpt : BPT, optional
            Precomputed Binary Partition Tree. If None and method is `"BPT"`,
            a new tree is built from `image_to_explain`.
        batch_size : int, default=64
            Number of masks evaluated per model call batch.
        verbose_plot : bool, default=False
            If True, plots intermediate coalition splits.
        pbar : tqdm progress bar, optional
            External progress bar object.
        min_area : int, default=1
            Minimum segment area. Segments with area <= `min_area` are not split.
        max_weight : float, optional
            Stop splitting coalitions whose recursive weight is below this value.

        Returns
        -------
        np.ndarray
            Attribution tensor with shape `(num_explained_classes, H, W)`.
        """
        assert min_area >= 1
        shap_values = np.zeros((self.num_explained_classes, 
                                self.image_to_explain.shape[0], 
                                self.image_to_explain.shape[1]))
        if method=='BPT':
            if bpt is None:
                bpt = build_bpt_from_image(self.image_to_explain)
            # assert bpt is not None, 'Expected argoment bpt='
            init_coalition = self.init_bpt(bpt)
        elif method=='AA':
            init_coalition = self.init_axisaligned()
        else:
            print('Unknown method', method) ; return None

        if self.verbose:
            pbar = pbar if pbar is not None else tqdm(total=max_evals, disable=False, leave=False)

        q = PriorityQueue()
        q.put(init_coalition)
        eval_count, reached_terminals = 0, 0
        while not q.empty():
            if eval_count >= max_evals: # no more v(s) budget
                while not q.empty():
                    coalition = q.get()
                    coalition.get_shapley(shap_values)
                break

            batch_masks = []
            batch_completers = []
            batch_owens = []
            while not q.empty() and len(batch_masks) < batch_size and \
                  eval_count + len(batch_masks) < max_evals:
                coalition = q.get()
                if (coalition.segment.area() <= min_area or
                    (max_weight is not None and coalition.weight<=max_weight)): 
                    reached_terminals += 1 # do not split further
                    coalition.get_shapley(shap_values)
                else:
                    (m_SuA, m_SuB, split_completer) = coalition.prepare_split(self)
                    batch_masks.append(m_SuA)
                    batch_masks.append(m_SuB)
                    batch_completers.append(split_completer)
                    batch_owens.append(coalition)

            if len(batch_masks) > 0:
                f = self.predict_masked(batch_masks)
                eval_count += len(batch_masks)
                if self.verbose: 
                    pbar.update(len(batch_masks))

                for i in range(len(batch_completers)):
                    f_SuA, f_SuB = f[i*2], f[i*2 + 1]
                    splits = batch_completers[i](f_SuA, f_SuB)
                    for o in splits:
                        q.put(o)
                    if verbose_plot:
                        fig,axes = plt.subplots(1, 5, figsize=(5,1))
                        batch_owens[i].plot(axes[0], self, alpha=0.5)
                        for i, s in enumerate(splits):
                            s.plot(axes[i+1], self)
                        plt.show()
        if self.verbose:
            pbar.refresh()
            if reached_terminals>0: 
                print(f'Reached {reached_terminals} terminals.')
        return shap_values

    def init_axisaligned(self):
        base = BaseSegment()
        s0 = AxisAlignedSegment(0, self.image_to_explain.shape[0],
                                0, self.image_to_explain.shape[1], base)
        return Coalition(self, s0, self.base_f_S, self.base_f_0, 1.0)

    def init_bpt(self, bpt):
        base = BaseSegment()
        s0 = BPT_Segment(bpt, bpt.N-1, base)
        return Coalition(self, s0, self.base_f_S, self.base_f_0, 1.0)


######################################################################################################

def plot_owen_values(explainer, shap_values, class_names, names=None):
    """
    Visualize ShapBPT explanations.

    Parameters
    ----------
    explainer : Explainer
        Fitted explanation object.

    shap_values : np.ndarray
        Explanation maps.

    class_names : list[str]
        Names of model output classes.

    names : list[str], optional
        Row labels for multiple explanation sets.

    Returns
    -------
    None
        Displays a matplotlib figure.
    """
    shap_values = np.array(shap_values)
    if len(shap_values.shape)==3: shap_values = np.array([shap_values])
    max_val = np.nanpercentile(np.abs(shap_values.flatten()), 99.9)
    num_explained_classes = len(explainer.base_f_S)
    num_rows = len(shap_values)
    fig,axes = plt.subplots(num_rows+1, num_explained_classes+1, 
                            figsize=(2*(num_explained_classes+1), 2*(num_rows+0.3)), 
                            squeeze=False,
                            height_ratios=[1]*num_rows + [0.3])
    base_image = explainer.image_to_explain
    if np.max(base_image)>1: base_image = base_image.astype(np.uint8)
    if len(base_image.shape)==2:
        base_image = np.stack([base_image, base_image, base_image], axis=-1)
    img_grey = (0.2989 * base_image[:, :, 0] +
                0.5870 * base_image[:, :, 1] + 
                0.1140 * base_image[:, :, 2])
    # axes[0].set_title(f'real: {class_names[expected_class]}')
    for r in range(num_rows):
        axes[r,0].imshow(base_image)
        for i in range(num_explained_classes):
            axes[r,i+1].imshow(img_grey.astype(base_image.dtype), alpha=0.50, cmap='gray')
            im=axes[r,i+1].imshow(shap_values[r,i], cmap=shapley_values_colormap, vmin = -max_val, vmax = max_val, alpha=0.80)
            if r==0: axes[r,i+1].set_title(f'{class_names[explainer.output_indexes[i]]}', fontsize=10)#+
                                #f'\n{explainer.base_f_S[i]:.5} to {explainer.base_f_0[i]:.5}')
        for jjj in range(num_explained_classes+1): axes[r,jjj].set_xticks([]) ; axes[r,jjj].set_yticks([])
    if names is not None:
        for r in range(num_rows):
            axes[r,0].set_ylabel(names[r])
    # Use the last row for the colorbar
    for ax in axes[-1,:]:
        ax.set_axis_off()
        # ax.set_box_aspect(0.1)
    cb = fig.colorbar(im, ax=axes[-1,:], label="Shapley/Owen value", 
                      orientation="horizontal", aspect=80, fraction=0.9)#, location='bottom') #,  fraction=0.5, 
    cb.outline.set_visible(False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # plt.tight_layout()
    plt.show()

######################################################################################################
