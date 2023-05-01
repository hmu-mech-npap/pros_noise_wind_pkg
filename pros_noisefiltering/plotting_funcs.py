"""A class to manage the plotting data for x, y axes."""
# %%

import matplotlib.pyplot as plt
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


def plot_response(fs: float, w: np.ndarray, h: np.ndarray, title: str):
    """Plot the gain frequency response of the filter based on the coeff.

    Args:
        fs (float): sampling frequency.
        w (np.ndarray): filter coefficients.
        h (np.ndarray): filter coefficients.
        title (str): title of plot.
    """
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    plt.ylim(-40, 5)
    # plt.xscale('log')
    # plt.xlim(0, 400)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title(title)


def plot_spectrum(x=np.ndarray,
                  y=np.ndarray,
                  title=str,
                  xlim=None):
    """Plot the power spectrum from the results of spect() function.

    Args:
        x (np.ndarray): frequencies calculated by Welch method (Hz).
        y (np.ndarray): the signal's power spectral magnitute.
        title (str): title of plot.
    """
    plt.figure()
    ax = plt.gca()
    ax.scatter(x, np.sqrt(y), s=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid(True, which='both')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitute')
    plt.title(title)

    try:
        plt.xlim(xlim)
    except:
        pass
