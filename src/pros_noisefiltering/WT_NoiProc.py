"""Main class for processing a tdms dataframe."""
# %%
# from pathlib import Path
# logging should go before matplotlib
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nptdms

from pros_noisefiltering.gen_functions import spect
from pros_noisefiltering.filters.iir import filt_butter_factory

from pros_noisefiltering.Graph_data_container import Graph_data_container

logging.basicConfig(level=logging.WARNING)

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
