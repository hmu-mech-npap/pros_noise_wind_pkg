"""Testing my functions for filtering and plotting FIR low pass filter."""
import pytest

import numpy as np
from pros_noisefiltering.WT_NoiProc import WT_NoiseChannelProc
from pros_noisefiltering.filters.iir import filt_butter_factory, apply_filter

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
def test_factory_iir(example_wtncp):
    """Test the fir factory method ensure proper operation."""
    iir_5_Hz = filt_butter_factory(filt_order=3, fc_Hz=5)
    filtered_signal = example_wtncp.filter(fc_Hz=5, filter_func=iir_5_Hz)

    assert isinstance(filtered_signal.data, np.ndarray)
    assert filtered_signal.data.shape is not None

    assert isinstance(iir_5_Hz.params, dict)
    assert iir_5_Hz.params is not None
    assert iir_5_Hz.params['fc_Hz'] == 5
    assert iir_5_Hz.params['filter order'] == 3
    return filtered_signal


def test_filtering(example_wtncp, test_factory_iir):
    """Testing the effects from filtering with IIR factory method."""
    assert "{:.1f}".format(
        test_factory_iir.data.mean()) < (
            "{:.1f}".format(
                example_wtncp.data.mean()))

    assert np.std(test_factory_iir.data) < np.std(example_wtncp.data)


def test_filters_apply_wrapper(example_wtncp):
    signal = apply_filter(example_wtncp.data, 1000)
    assert len(signal) is not None


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
