"""Main class for processing a tdms dataframe."""
# %%
# from pathlib import Path
# logging should go before matplotlib
import logging
from pathlib import Path
from matplotlib import pyplot as plt
import scipy.signal as signal
import numpy as np
import pandas as pd
import time
import re

import nptdms
# from nptdms import TdmsFile

import pyqtgraph as pg
import pyqtgraph.exporters
from pros_noisefiltering.gen_functions import spect, FFT_new
from pros_noisefiltering.Graph_data_container import Graph_data_container
logging.basicConfig(level=logging.WARNING)


# %% Functions and classes
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


filter_Butter_default = filt_butter_factory(filt_order=2, fc_Hz=100)


class WT_NoiseChannelProc():
    """Class for processing a tdms file for noise processing."""

    __measurement_history = []

    def __init__(self, desc: str, fs_Hz: float, data: np.ndarray,
                 channel_name: str, group_name: str, _channel_data,
                 operations: list):
        """Initialize constructor for managing tdms files."""
        self._channel_data = _channel_data
        self.set_description(desc=desc)
        # process details
        self.fs_Hz = fs_Hz
        self.data = data
        self.channel_name = channel_name
        self.group_name = group_name
        self.__measurement_history = operations

    def set_description(self, desc):
        """Set the description of the file.

        Args:
            desc (_type_): _description_
        """
        self.description = desc

    @property
    def operations(self):
        """Capture operations on data."""
        return self.__measurement_history

    @property
    def data_as_Series(self) -> pd.Series:
        """Return the raw data as a pd.Series.

        Returns:
            _type_: _description_
        """
        return pd.Series(self.data, name=f'{self.channel_name}:raw')

    def average(self, fr_Hz: float = 100, desc: str = None):
        """Return another object.

        Args:
        - fr_Hz (float) : recording frequency (with averaging)
        - desc (str) : description
            - Defaults to None: ie. add suffix _Av:<fr_Hz>)
        """
        ds = self.data_as_Series
        fs_Hz = self.fs_Hz

        ds_fr = ds.groupby(ds.index // (fs_Hz / fr_Hz)).mean().values

        description = (
            desc if desc is not None else self.description + f"_Av:{fr_Hz}")
        # TODO need to think better the naming conventions
        # (consider using a dictionary with multiple keys
        # e.g. different for plots etc)
        new_operations = self.operations.copy()
        new_operations.append(f'Apply Averaging :{fr_Hz}')

        return WT_NoiseChannelProc(desc=description,
                                   fs_Hz=fr_Hz, data=ds_fr,
                                   channel_name=self.channel_name,
                                   group_name=self.group_name,
                                   _channel_data=None,
                                   operations=new_operations
                                   )

    def set_desc(self, desc: str):
        """Set description for any operation or info for the data object."""
        return WT_NoiseChannelProc.from_obj(self, operation=None,
                                            desc=desc)

    def decimate(self, dec: int, offset: int = 0):
        """Return a decimated data seires.

        Args:
        - dec (int): decimation factor
        - offset (int): initial offset

        Returns:
        - WT_NoiseChannelProc.obj: a decimated data series.
        """
        decimated_fs_Hz = self.fs_Hz/dec

        new_operation = (
            f'Decimation factor:{dec}, Offset:{offset}, new fs_Hz:{decimated_fs_Hz}')

        return WT_NoiseChannelProc.from_obj(self,
                                            operation=new_operation,
                                            fs_Hz=decimated_fs_Hz,
                                            data=self.data[offset::dec])

    def _filter(self, fc_Hz: float,
                filter_func=filter_Butter_default,
                fs_Hz=None) -> pd.Series:
        """Return a filtered signal based on.

        Args:
            fc_Hz (float): cut off frequency in Hz
            filter_func (filt_butter_factory, optional): filtering function
                        that thates two arguments (ds, fs_Hz).
                        Defaults to 100, filt_order = 2).
            fs_Hz (None): sampling frequency in Hz (#TODO SHOULD BE REMOVED)


        Returns:
            _type_: _description_
        """
        if fs_Hz is not None:
            logging.warning(
                f'issued {fs_Hz} sampling frequency while the default is {self.fs_Hz}')
        fs_Hz = (
            fs_Hz if fs_Hz is not None else self.fs_Hz)

        filtered = filter_func(ds=self.data,
                               fs_Hz=fs_Hz,
                               fc_Hz=fc_Hz)
        return pd.Series(filtered,
                         name=f'{self.channel_name}:filt_fc_{fc_Hz}')

    def filter(self,  fc_Hz: float,
               filter_func=filter_Butter_default,
               fs_Hz=None, desc=None):
        """Filter a signal with the chosen factory method (IIR, FIR)."""
        ds_filt = self._filter(fc_Hz=fc_Hz,
                               filter_func=filter_func,
                               fs_Hz=fs_Hz)
        # description = (
        #     desc if desc is not None else self.description + f"_fc:{fc_Hz}")
        return WT_NoiseChannelProc.from_obj(self,
                                            data=ds_filt.values,
                                            operation=f'pass filter {fc_Hz}'
                                            )

    def calc_spectrum(self,
                      window='flattop', nperseg=1_024,
                      scaling='density') -> Graph_data_container:
        """Return a Graph_data_container object.

        with:
        - the power spectrum of the data.
        - uses the decimation function

        Returns:
        - classmethod: Graph_data_container
        """
        gobj = self.calc_spectrum_gen(dec=1, window=window,
                                      nperseg=nperseg, scaling=scaling)
        return gobj

    def calc_spectrum_gen(self, dec: int = 1,
                          offset: int = 0,
                          window='flattop',
                          nperseg=1_024,
                          scaling='density') -> Graph_data_container:
        """Create function for calculating the spectrum of a time series.

        Return a Graph_data_container object with the power spectrum
        of the **decimated** data.

        Args:
        - dec (int): decimation factor
        - offset (int, optional): offset.
            - Defaults to 0.

        Returns:
        - Graph_data_container: Containing the spectral density of the given
        signal.
        """
        decimated_fs_Hz = self.fs_Hz/dec
        x_r, y_r = spect(self.data[offset::dec],
                         FS=decimated_fs_Hz,
                         window=window,
                         nperseg=nperseg,
                         scaling=scaling)
        if dec == 1:
            label = f'{self.description}'
            # TODO the new chain method approach does not require this.
            # This is a remnant from the original approach.
        else:
            label = (
                f'{self.description}-{self.channel_name}-fs:{decimated_fs_Hz/1000:.1f}kHz ')

        return Graph_data_container(x=x_r, y=y_r,
                                    label=label)

    def plot_filtered_th(self, filter_func):
        """Plot the Time History.

        Args:
        - filter_func (factory method): Choose the wanted filter
        """
        plt.plot(self.data, label='raw')
        plt.plot(self.filter(filter_func=filter_func),
                 label=f'filtered: {filter_func.params}')

    @classmethod
    def from_tdms(cls, tdms_channel: nptdms.tdms.TdmsChannel, desc: str):
        """## Initiate factory method that generates an object class.

        #### TOD need to change the init function and test.

        Args:
        - tdms_channel (nptdms.tdms.TdmsChannel): tmds channel name
        - desc (str): descirption (used)
        """
        _channel_data = tdms_channel
        # desc

        fs_Hz = 1/_channel_data.properties['wf_increment']
        data = _channel_data.data
        channel_name = _channel_data.name
        group_name = _channel_data.group_name

        return cls(desc, fs_Hz, data, channel_name, group_name, _channel_data,
                   operations=[
                       f'Loaded from tdms file {group_name}/{channel_name}, {fs_Hz}'])

    @classmethod
    def from_obj(cls, obj, operation: str = None, **kwargs):
        """## Create factory method that generates an object class.

        ### If parameters are not provided the obj values are used.

        Args:
            tdms_channel (nptdms.tdms.TdmsChannel): tmds channel
            desc (str): descirption (used)
        """
        assert isinstance(operation, str) or None is None
        # the following line allows to pass a description maybe
        desc = obj.description if (
            kwargs.get('desc', None) is None) else (
                kwargs.get('desc', None))

        _channel_data = kwargs.get('_channel_data', obj._channel_data)
        fs_Hz = kwargs.get('fs_Hz', obj.fs_Hz)
        data = kwargs.get('data', obj.data)
        channel_name = kwargs.get('channel_name', obj.channel_name)
        group_name = kwargs.get('group_name', obj.group_name)

        new_ops = obj.operations.copy()
        if operation is not None:
            new_ops.append(operation)
        return cls(desc, fs_Hz, data, channel_name, group_name, _channel_data,
                   operations=new_ops)


# %%
class Plotter_Class():
    """# This is a class that can take different objects for plotting.

    Takes their raw data and plots:
    - a overview for the signal in frequency and time domains
    - Time histories (Not Implemented)
    - spectrums (Not Implemented)
    """

    # def __init__(self):
    #     """Reuse original self object."""
    #     super.__init__()

    @staticmethod
    def plot_signal_all_doms(signals, filt_func=filter_Butter_default,
                             export_only: bool = True):
        """## Plot a coprehensive and analytical figure with the signal info.

        Construct a GraphicsLayoutWidget and place many graphs in it.
        1. Raw and filtered signal in first row
            - filtering with
                - butterworth IIR filter
                - simple FIR filter

        2. frequency domain of the filtered fignal (fft)

        3. Spectral density of the filtered signal
            - with Welch's method

        Parameters
        ---
        - signals:
            - WT_NoiseChannelProc object
        - filt_func:
            - specifies the filtering constructor to be applied on the signal.
        - export_only (bool):
            - defines whether to save a .jpg file from the window.

        Returns
        ---
        - win:
            - GraphicsLayoutWidget.GraphicsLayoutWidget object
        - a "file".jpg (optional):
            - in ./sign_overview folder generated if not existing. For the file
        names a long string for analytical description is produced
        """
        for each in signals:
            filtrd = each.filter(fc_Hz=filt_func.params['fc_Hz'],
                                 filter_func=filt_func)

            # make a nice legend for filtered plot
            if filt_func.params['filter order'] > 25:
                filtrd.operations.append(
                    f'simple FIR low-pass {filt_func.params["fc_Hz"]} Hz')
            elif filt_func.params['filter order'] < 25:
                filtrd.operations.append(
                    f'butterworth IIR low-pass {filt_func.params["fc_Hz"]} Hz')

            freq_dom_filtrd = FFT_new(filtrd,
                                      title=
                                      f"Time domain filtered with {filtrd.description}")
            win = pg.GraphicsLayoutWidget(show=True,
                                          title="Basic plotting examples")

            filt_ops = filtrd.operations.pop(2)
            win.resize(1920, 1080)
            win.setWindowTitle('pyqtgraph example: Plotting')
            # Enable antialiasing for prettier plots
            pg.setConfigOptions(antialias=True)

            p1_raw = win.addPlot(row=0,
                                 col=0,
                                 colspan=1,
                                 title=f"{each.description}")
            p1_raw.setLabels(bottom='time duration (s)',
                             left='Raw sensor Voltage',)

            p1_raw.showGrid(y=True)
            p1_raw.plot(freq_dom_filtrd.time_sec,
                        each.data,
                        pen=(0, 255, 0, 35),
                        name="Raw signal")

            p1_filt = win.addPlot(row=0,
                                  col=1,
                                  title='Filtered signal')
            p1_filt.setLabels(bottom='time duration (s)', left='')
            p1_filt.showGrid(y=True)
            p1_filt.addLegend(labelTextSize='11pt')
            p1_filt.setYRange(filtrd.data.min() - 0.1,
                              filtrd.data.max() + 0.1)
            p1_filt.plot(freq_dom_filtrd.time_sec,
                         filtrd.data,
                         pen=(0, 0, 255),
                         name=filt_ops)

            p2_filt_fft = win.addPlot(row=1,
                                      col=0,
                                      rowspan=1,
                                      colspan=2,
                                      padding=10,
                                      title="Filtered signal Frequency domain representation")
            data = freq_dom_filtrd.fft_calc()
            p2_filt_fft.setLogMode(x=True, y=True)
            p2_filt_fft.showGrid(x=True, y=True)
            p2_filt_fft.setLabels(bottom='Frequencies in Hz',
                                  left='Power/Freq',
                                  top='')
            # p2.setLabel(axis='bottom', text='Frequencies in Hz')
            # p2.setLabel(axis='left', text='Power/Freq')
            p2_filt_fft.plot(data.x, data.y,
                             pen=(50, 50, 250),
                             fillLevel=-18,
                             brush=(250, 50, 50, 100))

            p3_filt_spect = win.addPlot(row=2,
                                        col=0,
                                        rowspan=1,
                                        colspan=2,
                                        padding=10,
                                        title="Filtered signal Spectral density (welch)")
            welch = filtrd.calc_spectrum_gen(nperseg=1024 << 6)
            p3_filt_spect.setLogMode(x=True, y=True)
            p3_filt_spect.showGrid(x=True, y=True)
            p3_filt_spect.setLabels(bottom='Frequencies in Hz', left='dB',
                                    top='')

            p3_filt_spect.plot(welch.x, welch.y,
                               pen=(50, 50, 250),
                               fillLevel=-18,
                               brush=(250, 50, 50, 100))

            if export_only is True:
                # time.sleep(2)
                exporter = pg.exporters.ImageExporter(win.scene())
                exporter.parameters()['width'] = 1920   # effects height
                exporter.parameters()['antialias'] = True

                # ensure normal operation on windows/linux
                my_path = Path('./sign_overview/')
                my_path.mkdir(parents=True, exist_ok=True)
                raw_name = f"{filt_ops}-{filtrd.description}"
                file_name = re.sub(r'=|-|\s', '_', raw_name)
                file_name = re.sub(r',|m/s', '', file_name)
                file_name = file_name.replace("()", "")

                formated_name = f'sign_overview/{file_name}.jpg'

                print(formated_name)
                exporter.export(formated_name)

            elif export_only is False:
                print("No exporting specified")
                pg.exec()


def plot_comparative_response(wt_obj,   # cutoff frequency
                              filter_func, response_offset=2e-4,
                              Kolmogorov_offset=1,
                              figsize=(16, 9),
                              nperseg=1024,
                              xlim=[1e1, 1e5],
                              ylim=[1e-8, 1e-1],
                              plot_th=False):
    """#TOD make this part of WT_NoiProc.

    Args:
    - wt_obj (Graph_data_container): List of graph containers
    - response_offset (float, optional): Setting a response offset.
        - Defaults to 2e-4.
    - figsize (tuple, optional): Determining the size of the produced figure.
        - Defaults to (16,9).
    """
    sig = wt_obj.data
    fs_Hz = wt_obj.fs_Hz

    filt_p = filter_func.params
    sos = signal.butter(N=filt_p['filter order'], Wn=filt_p['fc_Hz'],
                        btype='lp', fs=fs_Hz, output='sos')

    filtered = filter_func(sig, fs_Hz)

    # calculate spectrum
    f, Pxx_spec = signal.welch(sig, fs_Hz,
                               window='flattop', nperseg=nperseg,
                               scaling='density')
    f, Pxx_spec_filt = signal.welch(filtered, fs_Hz,
                                    window='flattop', nperseg=nperseg,
                                    scaling='density')

    # #TODO: this is for testing
    # Pxx_spec= Pxx_spec**2
    # Pxx_spec_filt = Pxx_spec_filt**2

    if plot_th:
        t = np.arange(0, len(sig), 1,
                      dtype='int') / fs_Hz
        # plot time domain
        fig1, (ax1, ax2) = plt.subplots(2, 1,
                                        sharex=True, sharey=True,
                                        figsize=figsize)
        fig1.suptitle(
            'Time Domain Filtering of signal with f1=10[Hz], f2=20[Hz] and noise')
        ax1.plot(t, sig)
        ax1.set_title('raw signal')
        # ax1.axis([0, 1, -2, 2])
        ax2.plot(t, filtered)
        ax2.set_title('After filter')
        # ax2.axis([0, 1, -2, 2])
        ax2.set_xlabel('Time [seconds]')
        plt.tight_layout()

    wb, hb = signal.sosfreqz(sos, worN=np.logspace(start=1, stop=5),
                             whole=True, fs=fs_Hz)
    fb = wb   # /(2*np.pi) # convert rad/s to Hz

    fig2, ax2 = plt.subplots(1, 1, sharex=True,
                             figsize=figsize)
    ax2.plot(fb,
             response_offset*abs(np.array(hb)), '--', lw=3,
             label='filter response')
    ax2.semilogy(f, np.sqrt(Pxx_spec), '.', label='raw')
    ax2.semilogy(f, np.sqrt(Pxx_spec_filt), '.', label='filtered')
    if Kolmogorov_offset is not None:
        KOLMOGORV_CONSTANT = - 5.0/3
        xs = f[1:]
        ys = xs**(KOLMOGORV_CONSTANT)*Kolmogorov_offset
        ax2.plot(xs, ys, 'r--', label='Kolmogorov -5/3')

    ax2.set_title('Filter frequency response (cutoff: {}Hz) -  {}'.format(
        filter_func.params.get('fc_Hz', None),
        wt_obj.description))
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Power Spectrum density [V**2/Hz]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.margins(0, 0.1)
    ax2.grid(which='both', axis='both')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.legend()
    # plt.savefig('Bessel Filter Freq Response.png')
# %%


# Define a class for FIR operations like
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
        # warmup = fir_filt_order-1
        # uncorrupted_output = sos_filt_data[warmup:]
        # filt_sig_time_int = time[warmup:]-((warmup/2)/fs_hz)
        # return uncorrupted_output           # uncorr_sos_output
        return sos_filt_data

    # Add the parameter attribute for checking filter response
    fir_filter.params = {'filter order': fir_order, 'fc_Hz': fc_Hz,
                         'filter type': 'simple fir'}
    return fir_filter


filter_fir_default = fir_factory_constructor(fir_order=2, fc_Hz=100)


# BUG i.e. replaced by the factory method for fir systems
# class Fir_filter:
