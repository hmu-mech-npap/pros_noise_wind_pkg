#%%[markdown]
#
#%%
from more_itertools import chunked
from scipy import signal
import numpy as np

def lp_firwin(numtaps_2:int, FS:float, cutoff_Hz:float):
    
    """
    Here the filter is a simple LP FIR filter
    

    Args:
        numtaps_2 (int): The order of the filter to be produced
        FS (float): The sampling frequency of the samples 
        cutoff_Hz (float): cutoff freq in Hz

    Returns:
        Filt(list):A list of arrays containing the filtered signal
        with no delay (time delay of FIR filter: time_delay= 1/2(numtaps-1))
        
        Blank(list):A list of arrays containing the filtered signal 
        with time delay from FIR filter process 

        TIME (np.ndarray):The time interval from the dataframe

    """

    fir_co = signal.firwin(numtaps_2, cutoff_Hz)
    w_fir_co, h_fir_co = signal.freqz(fir_co, [1])
    return fir_co, w_fir_co, h_fir_co
    

def filt_sig (coeff:np.ndarray, order:int, FS:float, Raw:list):
    """
    In this function the signal is filtered and the time delay
    of the FIR filter is rejected through the warmup process.
    The warmup operation is used to eliminate the filters delay.
    The delay of the FIR filter is producing corrupted samples 
    and the number of those samples are 1/2*(order-1). 
    {warmup}  

    Args:
        coeff (np.ndarray): the filter coefficients that will be used
        order (int): the order of the filter for the delay process
        FS (float): The sampling frequency of the signal
        Raw (list): The list of arrays from the dataframe of raw signal

    Returns: 
        Filt (list): A list of arrays containing the filtered signal
        with no delay (time delay of FIR filter: time_delay= 1/2(numtaps-1))
        
        Blank (list): A list of arrays containing the filtered signal 
        with time delay from FIR filter process 

        TIME (np.ndarray): The time interval from the dataframe
        TIME_NO_SHIFT (np.ndarray): The time interval of ther filtered signal with 
        no delay
    """
    #Filtering the raw signal with the above FIR filter 
    chunked_time=[]

    Blank = [] 
    x=[]
    for item in Raw:
        x=signal.lfilter(coeff, 1.0, item)
        Blank.append(x)

    #Time interval of the samples
    
    TIME = np.linspace(0, 7.599998, 3_800_000)
    chunked_time = TIME[::10]

    #The first N-1 samples are corrupted by the initial conditions
    warmup = order - 1

    #The phase delay of the filtered signal
    delay= (warmup / 2) / FS

    TIME_NO_SHIFT = chunked_time[warmup:]-delay


    #Uncorrupted signal
    Filt = []
    for item in Blank:
        Filt.append(item[warmup:])
    return Filt, Blank, chunked_time, TIME_NO_SHIFT