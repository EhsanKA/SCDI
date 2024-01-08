""" scDI - single cell Deep Interpretability """

from __future__ import print_function
from __future__ import division

from .methods import adata_top_genes, interpret, interpret_classes, remove_softmax, run_intMethods
from .plotting import bar, heatmap, scatterPlot, scplot, shorten_text, venn, venn2, venn2_circles, venn3, venn3_circles
from .attack import adversarial_attacks, adv_example
from .utils import *


from get_version import get_version
__version__ = get_version(__file__)
del get_version