"""Testing my functions for filtering and plotting FIR low pass filter."""
import pytest

import numpy as np
from pros_noisefiltering.WT_NoiProc import WT_NoiseChannelProc
from pros_noisefiltering.filters.fir import fir_factory_constructor

from pros_noisefiltering.Graph_data_container import Graph_data_container


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
        test_factory_fir.data.mean()) < (
            "{:.1f}".format(
                fir_wrapper.data.mean()))

    assert np.std(test_factory_fir.data) < np.std(fir_wrapper.data)



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
