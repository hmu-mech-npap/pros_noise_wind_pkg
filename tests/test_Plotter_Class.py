"""Testing plotter class methods for drawing figures for the signal."""
import pytest
from unittest.mock import patch
import numpy as np
from pros_noisefiltering.WT_NoiProc import (WT_NoiseChannelProc,
                                            fir_factory_constructor,
                                            Plotter_Class)


@pytest.fixture
def create_initial_signal() -> WT_NoiseChannelProc:
    """Create an object to plot in different domains."""
    return WT_NoiseChannelProc(desc='random signal speed = N/A',
                               fs_Hz=100_000,
                               # adding lots of samples to suppres nperseg warn
                               data=np.linspace(0, 100, 100_000),
                               channel_name='Torque',
                               group_name='Wind Measurement',
                               _channel_data=None,
                               operations=['New object (init)']
                               )


@pytest.fixture
@patch("pyqtgraph.plot")
def analyzer(mocker):
    """Create a mocking object for Plotter_Class.

    Using this technique we dont have to plot the graphs and show the window.
    Just mocking the object will give us a representation of what will happen
    if we call the function.
    """
    instance = mocker.MagikMock()
    return instance


@patch("pyqtgraph.plot")
def test_grapher_with_fir(mocker, create_initial_signal, analyzer):
    """Draw raw and filtered signals with fir filter."""
    hz_fir = fir_factory_constructor(fir_order=34, fc_Hz=1000)

    plot = analyzer.plot_signal_all_doms([create_initial_signal],
                                         filt_func=hz_fir)
    assert plot is not None


@patch("pyqtgraph.plot")
def test_grapher_with_butter(mocker, create_initial_signal, analyzer):
    """Draw with iir filter (default behavior)."""
    assert analyzer.plot_signal_all_doms([create_initial_signal]) is not None
