"""Tests for the main class for signal processing."""
import numpy as np
import pytest
from pros_noisefiltering.WT_NoiProc import WT_NoiseChannelProc

# def test_one():
#     pass


@pytest.fixture
def example_wtncp() -> WT_NoiseChannelProc:
    """Construct and object for testing the behavior of main class."""
    return WT_NoiseChannelProc(desc='description sample', fs_Hz=100,
                               data=np.zeros((100,)),
                               channel_name='Torque',
                               group_name='Wind Measurement',
                               _channel_data=None,
                               operations=['New object (init)']
                               )


def test_create_init(example_wtncp):
    """Assert the initial constructor."""
    assert example_wtncp.group_name == 'Wind Measurement'
    assert example_wtncp.channel_name == 'Torque'
    assert example_wtncp._channel_data is None
    np.testing.assert_equal(example_wtncp.data, np.zeros((100,)))
    assert example_wtncp.description == 'description sample'
    assert example_wtncp.fs_Hz == 100


def test_create_from_obj(example_wtncp):
    """Make a copy to test .from_object() method."""
    new_obj = WT_NoiseChannelProc.from_obj(example_wtncp,
                                           operation='copy')
    assert new_obj.group_name == 'Wind Measurement'
    assert new_obj.channel_name == 'Torque'
    assert new_obj._channel_data is None
    np.testing.assert_equal(new_obj.data, np.zeros((100,)))
    assert new_obj.description == 'description sample'
    assert new_obj.fs_Hz == 100
    assert len(new_obj.operations) == (len(example_wtncp.operations)+1)
    assert new_obj.operations[-1] == 'copy'


def test_decimate(example_wtncp):
    """Evaluate decimation operation."""
    new_obj = example_wtncp.decimate(dec=2)
    assert new_obj.group_name == 'Wind Measurement'
    assert new_obj.channel_name == 'Torque'
    assert new_obj._channel_data is None
    np.testing.assert_equal(new_obj.data, np.zeros((50,)))
    assert new_obj.description == 'description sample'
    assert new_obj.fs_Hz == 100/2
    assert len(new_obj.operations) == (len(example_wtncp.operations)+1)
    assert new_obj.operations[-1] == (
        'Decimation factor:2, Offset:0, new fs_Hz:50.0')


def test_filter(example_wtncp):
    """Evaluate the .filter() class method."""
    new_obj = example_wtncp.filter(fc_Hz=10)
    assert new_obj.group_name == 'Wind Measurement'
    assert new_obj.channel_name == 'Torque'
    assert new_obj._channel_data is None
    np.testing.assert_equal(new_obj.data, np.zeros((100,)))
    # assert new_obj.description == 'description sample_fc:10'
    assert new_obj.fs_Hz == 100
    assert len(new_obj.operations) == (len(example_wtncp.operations)+1)
    assert new_obj.operations[-1] == 'pass filter 10'
