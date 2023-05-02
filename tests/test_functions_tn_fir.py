"""Testing my functions for filtering and plotting FIR low pass filter."""
from unittest.mock import patch
import pytest
# Should be placed above unpublished pkg's
import numpy as np
from pros_noisefiltering.WT_NoiProc import (WT_NoiseChannelProc,
                                            filt_butter_factory,
                                            fir_factory_constructor,
                                            plot_comparative_response)
from pros_noisefiltering.gen_functions import (FFT_new,
                                               )
from pros_noisefiltering.plotting_funcs import Graph_data_container


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


@pytest.fixture
def fft_func(example_wtncp) -> WT_NoiseChannelProc:
    """Make a copy to ensure that copying works."""
    new_obj = WT_NoiseChannelProc.from_obj(example_wtncp, operation='copy')
    # filt = FFT_new(new_obj, title=None)
    return new_obj


def test_fft_info_init_func(fft_func):
    """Test the initially constructed object and attributes."""
    filt = FFT_new(fft_func, title='sample testing')

    assert isinstance(filt, FFT_new)
    assert filt.Title == 'sample testing'
    assert filt.fs_Hz == 1000
    assert len(filt.time_sec) == len(fft_func.data)
    assert filt._channel_data is None
    assert filt.operations is not None
    assert filt.group_name is not None
    assert filt.description is not None


def test_fft_calc_op(fft_func):
    """Test calculation of fft for a given signal."""
    fft_data = FFT_new(fft_func,
                       title=None).fft_calc()

    assert len(fft_data.x) == len(fft_data.y)
    assert fft_data.label == 'fft transform'
    assert fft_data.x.mean() > fft_func.data.mean()
    assert fft_data.y.mean() > fft_func.data.mean()
    assert isinstance(fft_data.xs_lim, list)
    assert np.std(fft_data.xs_lim) != np.std(fft_data.ys_lim)


@patch("matplotlib.pyplot.show")
def test_plot_operation(mock_show, fft_func):
    """Test plotting operation in my FFT class."""
    FFT_new(fft_func, title=None).fft_calc_and_plot()


@pytest.fixture
def fir_wrapper(example_wtncp):
    """Make a copy for testing the FIR methods."""
    obj_filt = WT_NoiseChannelProc.from_obj(example_wtncp, operation='copy')
    return (obj_filt)


@pytest.fixture
def test_factory_fir(fir_wrapper):
    """Test the fir factory method ensure proper operation."""
    fir_5_Hz = fir_factory_constructor(fir_order=30, fc_Hz=5)
    filtered_signal = fir_wrapper.filter(fc_Hz=5, filter_func=fir_5_Hz)

    assert isinstance(filtered_signal.data, np.ndarray)
    assert filtered_signal.data.shape is not None

    assert isinstance(fir_5_Hz.params, dict)
    assert fir_5_Hz.params is not None
    assert fir_5_Hz.params['fc_Hz'] == 5
    assert fir_5_Hz.params['filter order'] == 30
    return filtered_signal


def test_filtering(fir_wrapper, test_factory_fir):
    """Testing the effects from filtering with FIR factory method."""
    assert "{:.1f}".format(
        test_factory_fir.data.mean()) == (
            "{:.1f}".format(
                fir_wrapper.data.mean()))

    assert np.std(test_factory_fir.data) < np.std(fir_wrapper.data)


@patch("matplotlib.pyplot.show")
def test_fir_plots(mock_show, fir_wrapper):
    """Evaluate the FIR factory method for filters construction."""
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
    """Evaluate butterworth factory method and plotting operation."""
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
    """Copy object for testing classes."""
    return WT_NoiseChannelProc.from_obj(example_wtncp, operation="copy")


def test_construction_grapher(test_grapher) -> Graph_data_container:
    """Assert the construction of a Graph_data_container object."""
    for_plotting_data = Graph_data_container(x=test_grapher.data,
                                             y=test_grapher.data.mean,
                                             label='testing')
    assert for_plotting_data.x is not None
    assert for_plotting_data.y is not None
    assert for_plotting_data.label == 'testing'
