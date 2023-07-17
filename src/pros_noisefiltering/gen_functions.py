"""Generic functions for signal procassing."""
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pathlib

from pros_noisefiltering.Graph_data_container import Graph_data_container


# Define a function for Welch's method power spectrum for a signal
def spect(x: np.ndarray,
          FS: int,
          window='flattop',
          nperseg=1_024,
          scaling='spectrum'):
    """
    # Welch's method for power spectrum.

    Estimate the Power spectrum of a signal using the Welch method.

    Args:
    - x (np.ndarray):Column in dataframe managed as list of ndarrays.

    Returns:
    - z(np.ndarray): Array of sample frequencies
    - y(np.ndarray): Array of power spectrum of x
    """
    z, y = signal.welch(x,
                        FS,
                        window=window,
                        nperseg=nperseg,
                        scaling=scaling)
    return z, y

# Define a function for plotting the Power spectrums


def plot_spect_comb2(graph_objlist,
                     title: str,
                     alpha=1,
                     xlim=None,
                     ylim='auto',
                     Kolmogorov_offset=None,
                     markers=['.', 'o', 'x', '_'],
                     KOLMOGORV_CONSTANT=-5/3,
                     **kwargs
                     ):
    """## plots different signal power spectrums combined in one graph.

    This function plots the power spectrum diagram of an arbitray  signals.
    The amplitute and frequency of the signals are calculated with
    `signal.welch()` function.

    Args:
    - graph_objlist(list): a list of Graph_data_container
    - title (str): The main title of the figure
    - xlim (tuple): x limits of power spectrum graph
    """
    fig, ax = plt.subplots(1, 1,
                           figsize=kwargs.get('figsize',
                                              None))
    xylims = []
    no_grph = 0
    for gdc_obj in graph_objlist:
        assert isinstance(gdc_obj, Graph_data_container)
        marker = markers[no_grph % len(markers)]
        ax.scatter(gdc_obj.x, np.sqrt(gdc_obj.y),
                   label=f'{gdc_obj.label}',
                   s=kwargs.get('markersize', 2),
                   marker=marker,
                   alpha=alpha
                   )
        if kwargs.get('draw_lines', False):
            ax.plot(gdc_obj.x, np.sqrt(gdc_obj.y),
                    alpha=alpha
                    )
        xylims.append(gdc_obj.extrema)
        no_grph += 1

    try:
        plt.xlim(xlim)
    except:
        pass

    # ===========================  Plot Kolmoogov Line
    if Kolmogorov_offset is not None:
        xs = np.array(graph_objlist[0].xs_lim)
        ys = xs**(KOLMOGORV_CONSTANT)*Kolmogorov_offset
        ax.plot(xs, ys, 'r--', label='Kolmogorov -5/3')
    if ylim == 'auto':
        arr = np.array(xylims)[:, 2:]
        ax.set_ylim(np.sqrt([arr[:, 0].min(), arr[:, 1].max()]))

    elif isinstance(ylim, list):
        ax.set_ylim(ylim)
    # =================== final formating
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Spectral density [V**2/Hz]')
    # plt.legend(bbox_to_anchor=(1.04,0.5))
    ax.legend()
    ax.set_title(title)
    # ============================ save to disk
    if kwargs.get('to_disk', None) is True:
        # TODO remove this in favor of the fname (see below)
        target_path = pathlib.Path(
            '_temp_fig/{}.png'.format(
                title.translate({ord(c): None for c in ': /=\n'})))

        target_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target_path, facecolor='white', transparent=False)

    fname = kwargs.get('fname', None)
    if fname is not None:
        target_path = pathlib.Path(f'_temp_fig/{fname}.png')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target_path, facecolor='white', transparent=False)


class FFT_new:
    """This class is used to calculate the fourier transform for raw signal."""

    def __init__(self, signal, title):
        """Construct the appropriate object to manipulate the signal.

        We should be able to integrate this in WT_Noi_proc.
        """
        self.Title = title
        self.fs_Hz = signal.fs_Hz
        self.data = signal.data
        self.ind = signal.data_as_Series.index
        self.dt = 1 / int(self.fs_Hz)
        self.time_sec = self.ind * self.dt
        self.description = ''

        self._channel_data = signal._channel_data
        self.channel_name = signal.channel_name
        self.group_name = signal.group_name
        self.operations = signal.operations

    def fft_calc(self) -> Graph_data_container:
        """Calculate the frequency domain of raw data using fft algorithm.

        Uses the initial constructor to calculate the frequency domain of a
        given signal with constant sampling frequency.

        Args:
        - signal (WT_NoiseProcessor object): A given signal as the main class
        object for using internal methods such as `.operations` and `.data`.

        Returns:
        - Graph_data_container(x, y, label): Used to produce uniformal plots
        ammong the plotting functions.

        Usage:
        ```python
        # For calculating only and returning a Graph_data_container
        FFT_new(custom_object,
                title='Decimation number 1 INV INV ON').fft_calc()
        ```
        """
        n = len(self.time_sec)
        fhat = np.fft.fft(self.data, n)                  # compute fft
        PSD = fhat * np.conj(fhat) / n                  # Power spectrum (pr/f)
        freq = (1/(self.dt*n)) * np.arange(n)           # create x-axis (freqs)
        L = np.arange(1, np.floor(n/2), dtype=int)      # plot only first half
        return Graph_data_container(freq[L], abs(PSD[L]),
                                    label="fft transform")

    def fft_calc_and_plot(self):
        """Plot the frequency and time domain of the signal calculated above.

        Draw a figure with the time and frequency domain representation using
        `matplotlib` using (x, y) line plots.

        Usage:
        ```python
        # For drawing a simple graph
        FFT_new(custom_object,
                title='Decimation number 1 INV INV ON').fft_calc_and_plot()
        ```
        """
        fig, axs = plt.subplots(2, 1)

        plt.sca(axs[0])
        plt.grid('both')
        plt.title(self.Title)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitute (Voltage)')
        plt.plot(self.time_sec, self.data)
        # plt.loglog(freq[L],(PSD[L]))

        plt.sca(axs[1])
        plt.loglog(self.fft_calc().x, self.fft_calc().y)
        plt.title('Frequency domain')
        plt.xlabel('Frequencies [Hz]')
        plt.ylabel('Power/Freq')
        plt.grid('both')
        plt.show()
