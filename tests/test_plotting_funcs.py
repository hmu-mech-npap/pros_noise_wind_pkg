import numpy as np
import pytest
from unittest.mock import patch

from pros_noisefiltering.WT_NoiProc import WT_NoiseChannelProc
from pros_noisefiltering.plotting_funcs import plot_comparative_response

from pros_noisefiltering.filters.iir import filt_butter_factory
from pros_noisefiltering.filters.fir import fir_factory_constructor


@pytest.fixture
def example_wtncp() -> WT_NoiseChannelProc:
    """Construct a class object for testing."""
    return WT_NoiseChannelProc(desc='description sample', fs_Hz=1000,
                               # +15 seems to originate in bit flip
                               # from 0000 -> 1111 = 15
                               # needed to test the FIR functions with
                               # filtering
                               data=np.linspace(0, 100, 2000+15),
                               channel_name='Torque',
                               group_name='Wind Measurement',
                               _channel_data=None,
                               operations=['New object (init)']
                               )


@patch("matplotlib.pyplot.show")
def test_fir_plots(mock_show, example_wtncp):
    """Evaluate the FIR factory method for filters construction."""
    fc_Hz = 5
    fir_or = 30
    NPERSEG = 1_024
    fir_filter_cnstr_xorder = fir_factory_constructor(fir_order=fir_or,
                                                      fc_Hz=fc_Hz)
    FIGSIZE_SQR_L = (8, 10)
    plot_comparative_response(example_wtncp,       # cutoff frequency
                              filter_func=fir_filter_cnstr_xorder,
                              response_offset=2e-4,
                              Kolmogorov_offset=4e0,
                              nperseg=NPERSEG,
                              # xlim=0e0,
                              figsize=FIGSIZE_SQR_L)


@patch("matplotlib.pyplot.show")
def test_butter_plots(mock_show, example_wtncp):
    """Evaluate butterworth factory method and plotting operation."""
    NPERSEG = 1_024
    filter_Butter_default = filt_butter_factory(filt_order=2, fc_Hz=100)
    FIGSIZE_SQR_L = (8, 10)
    plot_comparative_response(example_wtncp,       # cutoff frequency
                              filter_func=filter_Butter_default,
                              response_offset=2e-4,
                              Kolmogorov_offset=4e0,
                              nperseg=NPERSEG,
                              # xlim=0e0,
                              figsize=FIGSIZE_SQR_L)
