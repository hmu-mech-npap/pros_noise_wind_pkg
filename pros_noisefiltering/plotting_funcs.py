#%%
from matplotlib import scale
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pathlib
import pandas as pd


class Graph_data_container:
    def __init__(self, x:np.ndarray, y:np.ndarray, label:str) -> None:
        self.x = x
        self.y = y
        self.label = label
    
    @property 
    def xs_lim(self):
        x_l =  np.floor(np.log10 (max(1, min(self.x)) ))
        x_u =  np.ceil(np.log10 (max(1, max(self.x)) ))
        return [10**x_l, 10**x_u]
    @property 
    def ys_lim(self):
        x_l =  np.floor(np.log10 ( min(self.y) ))-1
        x_u =  np.ceil(np.log10 ( max(self.y) ))+1
        return [10**x_l, 10**x_u]
    
    @property 
    def extrema(self):
        """returns the extreme values for x and y 
        [x_min, x_max, y_min, y_max]
        Returns:
            _type_: _description_
        """        
        return [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

    
def plot_response(fs:float, w:np.ndarray, h:np.ndarray, title:str):
    """Plots the gain frequency response of the filter based on the coefficients
    Args:
        fs (float): sampling frequency. 
        w (np.ndarray): filter coefficients.
        h (np.ndarray): filter coefficients.
        title (str): title of plot. 
    """
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    plt.ylim(-40, 5)
    #plt.xscale('log')
    #plt.xlim(0, 400)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title(title)


def plot_spectrum(x=np.ndarray,y=np.ndarray,
                title=str,xlim=None):
    """Plots the power spectrum from the results of spect() function
    Args:
        x (np.ndarray): frequencies calculated by Welch method (Hz).
        y (np.ndarray): the signal's power spectral magnitute from Welch method ().
        title (str): title of plot.
    """
    plt.figure()
    ax=plt.gca()
    ax.scatter(x, np.sqrt(y), s=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid(True, which='both')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitute')
    plt.title(title)
    
    try:
        plt.xlim(xlim)
    except:
        pass

# Plot two spectrum diagrams of 2 different signals
# Should be replaced with plot_spect_comb2 function because 
# is more flexible than plot_spect_comb
#
# def plot_spect_comb(x1:np.ndarray,y1:np.ndarray,
#                     x2:np.ndarray,y2:np.ndarray,
#                     x3:np.ndarray, y3:np.ndarray,
#                     title:str, slabel1:str,slabel2:str,slabel3:str,
#                     xlim = None):
#     """ ## Three signals power spectrums combined in one graph
#     This function plots the power spectrum diagram of two signals.
#     The amplitute and frequency of the signals are calculated with signal.welch() function.
#     Args:
#         x1 (np.ndarray): The frequencies of the first given signal to be plotted
#         y1 (np.ndarray): The amplitute of the first given signal to be plotted
#         x2 (np.ndarray): The frequencies of the second given signal to be plotted
#         y2 (np.ndarray): The amplitute of the second given signal to be plotted
#         x3 (np.ndarray): The frequencies of the third given signal to be plotted
#         y3 (np.ndarray): The amplitute of the third given signal to be plotted
#         title (str): The main title of the figure 
#         slabel1 (str): The label to be presented in the legend of the plot for the first signal
#         slabel2 (str): The label to be presented in the legend of the plot for the second signal
#         slabel2 (str): The label to be presented in the legend of the plot for the third signal
#     """
#     plt.figure()
#     ax = plt.gca()
#     ax.scatter(x1, np.sqrt(y1), label=f'{slabel1}', s=1)
#     ax.scatter(x2, np.sqrt(y2), label=f'{slabel2}', s=1)
#     ax.scatter(x3, np.sqrt(y3), label=f'{slabel3}', s=1)
# 
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     plt.grid(True, which='both')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Amplitute')
#     plt.legend(bbox_to_anchor=(1.04,0.5))
#     plt.title(title)
#     
#     try:
#         plt.xlim(xlim)
#     except:
#         pass
 
 