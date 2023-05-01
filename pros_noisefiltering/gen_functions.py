#%%
from matplotlib import scale
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pathlib
import pandas as pd

from pros_noisefiltering.Graph_data_container import Graph_data_container
# FS=500_000  #Sampling frequency in Hz 

# Define function for Filter freq response


#Define a function for Welch's method power spectrum for a signal

def spect (x:np.ndarray, FS:int, window='flattop', nperseg=1_024, scaling='spectrum'):
    """
    # Welch's method for power spectrum
    
    Estimate the Power spectrum of a signal using the Welch method
    Args:
        x (np.ndarray):Column in dataframe managed as list of ndarrays.
    Returns:
       z(np.ndarray): Array of sample frequencies
       y(np.ndarray): Array of power spectrum of x
    """
    z,y = signal.welch(x,FS,window=window, nperseg=nperseg, scaling=scaling)
    return z, y

#Define a function for plotting the Power spectrums


def plot_spect_comb2(graph_objlist ,
                    title:str, 
                    alpha = 1,
                    xlim = None, 
                    ylim = 'auto',
                    Kolmogorov_offset = None,
                    markers = ['.','o','x','_'],
                    KOLMOGORV_CONSTANT = - 5/3,
                    **kwargs
                    ):
    """ ## plots different signal power spectrums combined in one graph
    This function plots the power spectrum diagram of an arbitray  signals.
    The amplitute and frequency of the signals are calculated with signal.welch() function.
    Args:
        graph_objlist(list): a list of Graph_data_container
        title (str): The main title of the figure 
        xlim (tuple): x limits of power spectrum graph
    """

    fig, ax = plt.subplots(1,1, figsize=kwargs.get('figsize',None))
    xylims = []
    no_grph = 0
    for gdc_obj in graph_objlist:
        assert isinstance(gdc_obj, Graph_data_container)
        marker = markers[no_grph% len(markers)]
        ax.scatter(gdc_obj.x, np.sqrt(gdc_obj.y), 
                   label=f'{gdc_obj.label}', 
                   s=kwargs.get('markersize',2),
                   marker=marker,
                   alpha=alpha
                   )
        if kwargs.get('draw_lines',False):
            ax.plot(gdc_obj.x, np.sqrt(gdc_obj.y), 
                   alpha=alpha
                   )
        xylims.append(gdc_obj.extrema)
        no_grph +=1
    
    
    try:
        plt.xlim(xlim)
    except:
        pass
    
    # ===========================  Plot Kolmoogov Line
    if Kolmogorov_offset is not None:
        xs = np.array(graph_objlist[0].xs_lim)
        ys = xs**(KOLMOGORV_CONSTANT)*Kolmogorov_offset
        ax.plot(xs,ys, 'r--', label = 'Kolmogorov -5/3')
    if ylim == 'auto':
        arr= np.array(xylims)[:,2:]
        ax.set_ylim(np.sqrt([arr[:,0].min(), arr[:,1].max() ]))
        
    elif isinstance(ylim, list):
        ax.set_ylim( ylim)
    # =================== final formating
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Spectral density [V**2/Hz]')
    # plt.legend(bbox_to_anchor=(1.04,0.5))
    ax.legend()
    ax.set_title(title)
    # ============================ save to disk
    if kwargs.get('to_disk', None) is True:
        #TODO remove this in favor of the fname (see below) 
        target_path = pathlib.Path('_temp_fig/{}.png'.format(title.translate({ord(c): None for c in ': /=\n'} )))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target_path,facecolor='white', transparent=False)

    fname = kwargs.get('fname', None)
    if fname is not None:
        target_path = pathlib.Path(f'_temp_fig/{fname}.png')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target_path,facecolor='white', transparent=False)
    
#%%    
# New classes for storing signal and titles information for human readable code 
# and faster implementation regardless the actual operation of a function
class Axis_titles:
    def __init__(self, x_title:str, y_title:str) -> None:
        self.x_title = x_title
        self.y_title = y_title

class Time_domain_data_cont():
    """A class for importing the x,y variables for a signal in time domain
    to plotting functions.
    """

    def __init__(self,x:np.ndarray,y:np.ndarray,label:str) -> None:
        """_summary_

        Args:
            x (np.ndarray): time duration of the signal in seconds.
            y (np.ndarray): amplitute in dB (decibel)
            label (str): signal information for the figure legend
        """
        self.x = x
        self.y = y
        self.label = label
     

#Define function to plot the raw and filtered signals combined 

def plot_signals(time_domain_sig,
                axis_titles:str,
                Title:str,
                **kwargs,
                ):
    """ ## Plot signals in time domain
    This function is used to plot signals in time domain from the old dataset.
    It was updated to use class objects as import for the x,y components of the signal and
    the axis titles instead of a variable oriented import (old plot_signals function)
    which was more static and ram consuming.


    Args:
        time_domain_sig (list): A list created from Time_domain_data_cont
        axis_titles (str): The axis titles
        Title (str): The titles to be plotted in each figure

    """
    fig, ax = plt.subplots(1,1, figsize=kwargs.get('figsize',None))
    for obj in time_domain_sig:
        ax.scatter(obj.x,obj.y, label=f'{obj.label}', s=1)

    for ax_title in axis_titles:
        ax.set_title(Title)
        ax.set_ylabel(ax_title.y_title)
        ax.set_xlabel(ax_title.x_title)
        ax.grid(True, which='both')
        ax.legend(bbox_to_anchor=(1.04,0.5))

class Signals_for_fft_plot:
    def __init__(self,freq,sig1:np.ndarray,sig2:np.ndarray) -> None:
        self.raw_mag = sig1
        self.filt_mag = sig2
        self.x = freq

class fft_calc_sig:
    def __init__(self, x1: np.ndarray, x2: np.ndarray, label: str) -> None:
        self.x1 = x1
        self.x2 = x2

#Define function for FFT of two signals to be able of plotting
#the corrupted and uncorrupted signals in frequency domain
def fft_sig (signals,    f0 = 2_000,fs = 500_000):
    """
    Computes the fourier transform for two seperate signals and returns the results
    and the corresponding frequencies

    Args:
        y1 (np.ndarray): array object corresponding to raw signal.
        y2 (np.ndarray): array object corresponding to filtered signal.

    Returns:
       f_plot(np.ndarray): array_like of sample frequencies
       y_input_mag_plot(np.ndarray): Amplitude of raw signal samples
       y_output_mag_plot(np.ndarray): Amplitude of filtered signal samples

    """

    N = int(2_000*(fs/f0)) #TODO what is this N???
    # This N was used in a video which i found on Youtube for computing and
    # plotting the FFT of a signal.
    # Even the guy in the video was not sure why to use this N but
    # fs is the sampling frequency and f0 is the signal frequency.
    # I multiply with 2_000 to eliminate the signal frequency and use it as
    # the sampling frequency for plotting the signal in freq domain (x axis).
    # #TODO have to find something better couse it's a black box for me
    # and the source is unreliable

    f= np.linspace (0, (fs-1), fs)
    for element in signals:

        yf_input = np.fft.fft(element.x1)
        y_input_mag_plot = np.abs(yf_input)/fs
        f_plot = f[0:int(fs/2+1)]
        y_input_mag_plot = 2*y_input_mag_plot[0:int(fs/2+1)]
        y_input_mag_plot[0] = y_input_mag_plot[0] / 2

        yf_output = np.fft.fft(element.x2)
        y_output_mag_plot = np.abs(yf_output)/fs
        y_output_mag_plot = 2* y_output_mag_plot[0:int(fs/2+1)]
        y_output_mag_plot[0]= y_output_mag_plot[0]/2


    # dt = 0.01              # time interval
    # t=np.arange(0,7.5,dt)     # time array
    # n=len(t)                # number of samples
    # fhat = np.fft.fft(y1,n) # compute the fft for the first signal
    # PSD = fhat * np.conj(fhat) / n
    # freq = (1/(dt*n)) * np.arange(n)
    # L = np.arange(1,np.floor(n/2), dtype='int')


    return(f_plot, y_input_mag_plot, y_output_mag_plot,
            #freq, L, PSD
            )


#Define function for the FFT plot
# This function will be written to use class objects for more ease of use in 
# importing and not to be bound to variables for less RAM usage
#  
class Fft_Plot_info:
    def __init__(self, Title:list, filter_type:str, signal_state:str) -> None:
        """Initiate a class for importing information used in fft graph

        Args:
            Title (list): The titles to be presented in each graph
            explaining the device's configuration for each measurement.

    ### Filtering process information:
    #### Figure label information for plotting in output graph

        filter_type (str): The filter type used to produce the output

        signal_state (str): This defines if there is a corruption during the
        filtering process
        """
        self.title = Title
        self.filt_type = filter_type
        self.sig_state = signal_state

#Define function for the FFT plot
def plot_FFT (signals,
                info,
                axis_titles:str,
                **kwargs
                ):
    """
    Function for plotting the raw and filtered signals in
    frequency domain. On the x axis we plot the frequency in Hz
    and on the y axis the amplitude of the sample at that frequency

    Args:
        signals (_type_): Class object with the raw signal
        info (_type_): Information for the legend
        axis_titles (str): X,Y axis titles
    """

    for objct,obj,ax_titles in zip(signals, info, axis_titles):
        fig, (ax1, ax2) = plt.subplots(2, 1,
                                        sharex=True, sharey=True,
                                        figsize=kwargs.get('figsize',None))

        fig.suptitle(obj.title)
        ax1.loglog(objct.x, objct.raw_mag)
        ax1.grid(True, which = 'both')
        ax1.set_ylabel(ax_titles.y_title)
        ax2.loglog(objct.x, objct.filt_mag, 'orange',
                    label = f'{obj.sig_state} {obj.filt_type} output')

        ax2.grid(True, which='both')
        ax2.set_xlabel(ax_titles.x_title)
        ax2.set_ylabel(ax_titles.y_title)
        plt.legend()
        plt.show()

# Adding WT_Noise_ChannelProcessor to use the signal info from nptdms 
# TODO remove this (DUPlICATE)  move to filters
def apply_filter(ds:np.ndarray, fs_Hz:float, fc_Hz = 100, filt_order = 2 ):
                 # filter cutoff frequency
    sos = signal.butter(filt_order , fc_Hz, 'lp', fs=fs_Hz, output='sos')
    filtered = signal.sosfilt(sos, ds-ds[0])+ds[0]
    return filtered



# %%

def data_import(file_path:str, file_name_of_raw:str):
    """file import script and data chunking for faster overall process time

    Args:
        file_path (str): the file path of the folder containing the HDF5 file
        file_name_of_raw (str): the name of the file to import
    Returns:
        MATRIX_RAW (list): a list of np.ndarrays transformed columns from dataset
        L (list): a list of the dataframes keys in str format
        List_of_chunked (list): Here we store the sampled raw
        signal from the dataset with a constant rate 

    """

    # #Read and store the .h5 file with pandas
    # f_1 = pd.HDFStore(path='/home/goodvibrations32/Documents/Git_clones_dissertation/DSP_Filters_Python-/src/data_folder/noise_reference_raw.h5', mode='r')
    # 
    # #Store all the dataframe in a variable
    # data_raw = f_1['/df']
    # 

    #file_path = input('The full path of raw data file to process: ' )
    #file_name_of_raw =input('Enter the name of the raw signal file :') 

    #Read and store the .h5 file with pandas
    f_1 = pd.HDFStore(path=f'{file_path}{file_name_of_raw}', mode='r')

    print('The data frame key is: ',f_1.keys())


    data_raw = f_1['df']

    #Chunking of data with a constant sample rate
    rate_of_sampling = 10

    chunked_data = data_raw[::rate_of_sampling]
    List_of_chunked = []

    #Make a list with present keys
    L = list(data_raw.keys())

    #Store the chunked data in a list for the signal processing operations
    for element1 in L:
        List_of_chunked.append(np.array(chunked_data.get(element1)))
    print(data_raw.info())

    #Manage data with lists 
    MATRIX_RAW = []
    for element0 in L:
        MATRIX_RAW.append(np.array(data_raw.get(element0)))
    
    return MATRIX_RAW, L, List_of_chunked, file_path, file_name_of_raw


class FFT_new:
    """This class is used to calculate the fourier transform for raw signal."""

    def __init__(self, signal, title):
        """Construct the appropriate object to manipulate the signal.

        We should be able to integrate this in WT_Noi_proc.
        """
        self.Title = title
        self.sr = signal.fs_Hz
        self.sig = signal.data
        self.ind = signal.data_as_Series.index
        self.dt = 1 / int(self.sr)
        self.time_sec = self.ind * self.dt

        self._channel_data = signal._channel_data

    def fft_calc(self):
        """Func body for calculation of the frequency domain of raw data."""
        n = len(self.time_sec)
        fhat = np.fft.fft(self.sig, n)                  # compute fft
        PSD = fhat * np.conj(fhat) / n                  # Power spectrum (pr/f)
        freq = (1/(self.dt*n)) * np.arange(n)           # create x-axis (freqs)
        L = np.arange(1, np.floor(n/2), dtype=int)      # plot only first half
        return Graph_data_container(freq[L], abs(PSD[L]), label="fft transform")

    def fft_calc_and_plot(self):
        """Func body for calculation of the frequency domain of raw data."""
        n = len(self.time_sec)
        fhat = np.fft.fft(self.sig, n)                  # compute fft
        PSD = fhat * np.conj(fhat) / n                  # Power spectrum (pr/f)
        freq = (1/(self.dt*n)) * np.arange(n)           # create x-axis (freqs)
        L = np.arange(1, np.floor(n/2), dtype=int)      # plot only first half

        fig, axs = plt.subplots(2, 1)

        plt.sca(axs[0])
        plt.grid('both')
        plt.title(self.Title)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitute (Voltage)')
        plt.plot(self.time_sec, self.sig)
        # plt.loglog(freq[L],(PSD[L]))

        plt.sca(axs[1])
        plt.loglog(freq[L], abs(PSD[L]))
        plt.title('Frequency domain')
        plt.xlabel('Frequencies [Hz]')
        plt.ylabel('Power/Freq')
        plt.grid('both')
        plt.show()
