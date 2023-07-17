"""Testing my functions for general operations such as FFT and more."""
from unittest.mock import patch
import pytest
import pathlib
import numpy as np

from pros_noisefiltering.WT_NoiProc import WT_NoiseChannelProc
from pros_noisefiltering.gen_functions import FFT_new, spect, plot_spect_comb2


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


def test_spect_func(example_wtncp):
    X_r, Y_r = spect(example_wtncp.data, FS=100)
    assert (X_r is not None)
    assert (Y_r is not None)


@patch("matplotlib.pyplot.show")
def test_plot_spect_comb2(mock_show, example_wtncp):
    plot_spect_comb2([FFT_new(example_wtncp, title=None).fft_calc()],
                     title=None,
                     alpha=0.5,
                     fname="dummy",
                     to_disk=False,
                     Kolmogorov_offset=None,)
    assert pathlib.Path("./dummy/") is not None



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
    assert filt.ind is not None
    assert filt.dt == 1/fft_func.fs_Hz
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

