from unittest.mock import patch
import pytest
from pros_noisefiltering.WT_NoiProc import (WT_NoiseChannelProc,
                                            filt_butter_factory,
                                            fir_factory_constructor,
                                            plot_comparative_response)
from pros_noisefiltering.gen_functions import (FFT_new,
                                               )
from pros_noisefiltering.plotting_funcs import Graph_data_container
import numpy as np


@pytest.fixture
def example_wtncp() -> WT_NoiseChannelProc:
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


@pytest.fixture
def fft_func(example_wtncp) -> WT_NoiseChannelProc:
    new_obj = WT_NoiseChannelProc.from_obj(example_wtncp, operation='copy')
    # filt = FFT_new(new_obj, title=None)
    return(new_obj)


def test_fft_info_init_func(fft_func):
    filt = FFT_new(fft_func, title=None)
    assert filt.sr == 1000
    assert len(filt.time_sec) == len(fft_func.data)
    assert fft_func._channel_data is None
    assert fft_func.operations is not None


@patch("matplotlib.pyplot.show")
def test_plot_operation(mock_show, fft_func):
    FFT_new(fft_func, title=None).fft_calc_and_plot()


@pytest.fixture
def fir_wrapper(example_wtncp):
    obj_filt = WT_NoiseChannelProc.from_obj(example_wtncp, operation='copy')
    return (obj_filt)


@patch("matplotlib.pyplot.show")
def test_fir_plots(mock_show, fir_wrapper):
    fc_Hz = 5
    fir_or = 30
    NPERSEG = 1_024
    fir_filter_cnstr_xorder = fir_factory_constructor(fir_order=fir_or,
                                                      fc_Hz=fc_Hz)
    FIGSIZE_SQR_L = (8, 10)
    plot_comparative_response(fir_wrapper,       # cutoff frequency
                              filter_func=fir_filter_cnstr_xorder,
                              response_offset=2e-4,
                              Kolmogorov_offset=4e0,
                              nperseg=NPERSEG,
                              # xlim=0e0,
                              figsize=FIGSIZE_SQR_L)


@patch("matplotlib.pyplot.show")
def test_butter_plots(mock_show, fir_wrapper):
    NPERSEG = 1_024
    filter_Butter_default = filt_butter_factory(filt_order=2, fc_Hz=100)
    FIGSIZE_SQR_L = (8, 10)
    plot_comparative_response(fir_wrapper,       # cutoff frequency
                              filter_func=filter_Butter_default,
                              response_offset=2e-4,
                              Kolmogorov_offset=4e0,
                              nperseg=NPERSEG,
                              # xlim=0e0,
                              figsize=FIGSIZE_SQR_L)


@pytest.fixture
def test_grapher(example_wtncp):
    return WT_NoiseChannelProc.from_obj(example_wtncp, operation="copy")


def test_construction_grapher(test_grapher) -> Graph_data_container:
    for_plotting_data = Graph_data_container(x=test_grapher.data,
                                             y=test_grapher.data.mean,
                                             label='testing')
    assert for_plotting_data.x is not None
    assert for_plotting_data.y is not None
    assert for_plotting_data.label == 'testing'
