import scipy.signal as signal
import numpy as np


def apply_filter(ds: np.ndarray, fs_Hz: float, fc_Hz=100, filt_order=2):
    """Apply a butterworth IIR filter."""
    sos = signal.butter(filt_order, fc_Hz, 'lp', fs=fs_Hz, output='sos')
    filtered = signal.sosfilt(sos, ds-ds[0])+ds[0]
    return filtered


# Functions and classes
def filt_butter_factory(filt_order=2, fc_Hz: float = 100):
    """Construct a factory method that produces a BUTTERWORTH filter function.

    Here we construct a butterworth IIR filter with a filter order and
    a cutoff frequency passed as arguments. The function returns a filter using
    `scipy.signal` module to generate the it.

    Args:
    - filt_order (int, optional)  : Filter order. Defaults to 2.
    - fc_Hz (float, optional)     : cut off frequency in Hz (defaults to 100)
    """
    def filt_butter(ds: np.ndarray,
                    fs_Hz: float, fc_Hz: float = fc_Hz,
                    filt_order=filt_order):
        """Apply filtering to np.array with fs_Hz data, with cuffoff frequency.

        Args:
        - ds (np.ndarray): data signal object
        - fs_Hz (float): sampling frequency
        - fc_Hz (int, optional): cutoff frequency of filter(low pass def:100)
        - filt_order (int, optional): The order of the filter (number of
        coeff.) to be produced. Defaults to 2. def:2

        Returns:
        - filtered (signal.sosfilt()): The filter with cutoff and order defined
        as arguments
        """
        sos = signal.butter(filt_order, fc_Hz,
                            'lp', fs=fs_Hz,
                            output='sos')
        filtered = signal.sosfilt(sos, ds-ds[0])+ds[0]
        return filtered
    # additional decoration sith params dictionary
    # this is used instead of a class
    filt_butter.params = {'filter order': filt_order, 'fc_Hz': fc_Hz}
    return filt_butter
