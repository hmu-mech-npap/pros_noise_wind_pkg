"""A class to manage the plotting data for x, y axes."""
# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import pyqtgraph.exporters
import pyqtgraph as pg
import re

from pros_noisefiltering.gen_functions import FFT_new
from pros_noisefiltering.filters.iir import filt_butter_factory
from pros_noisefiltering.filters.fir import fir_factory_constructor

filter_fir_default = fir_factory_constructor(fir_order=2, fc_Hz=100)
filter_Butter_default = filt_butter_factory(filt_order=2, fc_Hz=100)


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
                                      title=f"Time domain filtered with {filtrd.description}")
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
