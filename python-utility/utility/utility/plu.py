"""Matplotlib plotting operations"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as clr
import matplotlib.collections as mc
import matplotlib.cm as cm
import matplotlib.patches as patches

def modify_alpha(c, alpha):
    """Create a RGBA tuple of the color with modified alpha.
    Example: ('red', 0.2) -> (1.0, 0, 0, 0.2)."""
    tup = list(clr.to_rgba(c))
    tup[3] = alpha
    return tuple(tup)