"""Functions for simple FIR filter construction."""

# from more_itertools import chunked
from scipy import signal
import numpy as np


def fir_factory_constructor(fir_order=32, fc_Hz: float = 200):
    """Mimicing so this is working with Papadakis solution above.

    Description
    -----------
    Now we can use the `.filter()` class method to filter with a simple
    low-pass `FIR`. The idea is to be able to construct tables for standard
    deviation comparison of the methods. This can be also used to define the
    needed fir filter and plot the results with:
      - `plot_comparative_response(wt_obj, filter_func)`
      - wherever we use the `filter_func` keyword to select the filtering
        function.


    Usage
    ------
    ```python
    from pros_noisefiltering.WT_NoiProc import fir_factory_constructor

    fir_200 = fir_factory_constructor(fir_order=60, fc_Hz=200)
    fir_filt_200 = ca1_0.filter(fc_Hz=2_00, filter_func=fir_200).data
    ```
    """
    def fir_filter(ds: np.ndarray,
                   fs_Hz: float, fc_Hz: float = fc_Hz,
                   fir_filt_order=fir_order):

        fir_filt_coeff = signal.firwin(numtaps=fir_filt_order,
                                       fs=fs_Hz,
                                       cutoff=fc_Hz,
                                       # pass_zero=False ,
                                       # scale= True,
                                       )
        # # Hann approach
        # fir_filt_coeff=signal.firwin(fir_order + 1,
        #                              [0, 200/fs_hz],
        #                              fs=fs_hz , window='hann')

        # make output sos type to ensure normal operation
        # this is crusial for elimination of ending ripples see image above
        sos_fir_mode = signal.tf2sos(fir_filt_coeff, 1)
        sos_filt_data = signal.sosfilt(sos_fir_mode, ds-ds[0])+ds[0]

        return sos_filt_data

    # Add the parameter attribute for checking filter response
    fir_filter.params = {'filter order': fir_order, 'fc_Hz': fc_Hz,
                         'filter type': 'simple fir'}
    return fir_filter
