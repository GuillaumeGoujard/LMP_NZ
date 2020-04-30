import numpy as np
import pandas as pd

def add_arrow(line, position=None, direction=1, size=50, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = pd.Series(line.get_xdata())
    ydata = pd.Series(line.get_ydata())

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = 1/2 * (1 - direction)
    end_ind = 1/2 * (1 + direction)

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=((xdata[end_ind]+xdata[start_ind])/2, (ydata[start_ind]+ydata[end_ind])/2),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )