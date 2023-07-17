"""A class to manage the plotting data for x, y axes."""
import numpy as np


class Graph_data_container:
    """Managing plots to appear uniformal and cohesive."""

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 label: str) -> None:
        """Initialize constructor for plotting data."""
        self.x = x
        self.y = y
        self.label = label

    @property
    def xs_lim(self):
        """Limit x axis length for expresive plots."""
        x_l = np.floor(np.log10(max(1, min(self.x))))
        x_u = np.ceil(np.log10(max(1, max(self.x))))
        return [10**x_l, 10**x_u]

    @property
    def ys_lim(self):
        """Limit y axis length like x axis."""
        x_l = np.floor(np.log10(min(self.y)))-1
        x_u = np.ceil(np.log10(max(self.y)))+1
        return [10**x_l, 10**x_u]

    @property
    def extrema(self):
        """Return the extreme values for x and y.

        [x_min, x_max, y_min, y_max]
        Returns:
            _type_: _description_
        """
        return [self.x.min(), self.x.max(), self.y.min(), self.y.max()]
