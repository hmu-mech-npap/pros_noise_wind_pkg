# npp_wtblade: Processing of Wind turbine data 

This is a package for helping with the processing of Wind Tunnel and Wind Blade measurements. 

Author: Nikolaos Papadakis, N. Torosian

## installation

```console
cd path/to/pkg
```
and then 
```console
python setup.py install
```
## usage

to use it import like normal:

``` python
import pros_noisefiltering
# OR
from pros_noisefiltering import *
```

## basic functionality

### using the main class `WT_NoiseChannelProc()` and `plot_spect_comb2()`

``` python
from pros_noisefiltering.WT_NoiProc import WT_NoiseChannelProc
from pathlib import Path
from nptdms import TdmsFile

GROUP_NAME = 'Wind Measurement'
CHAN_NAME = 'Wind2'

tdms_raw_WT = TDMSFile(Path('path/to/data.tdms'))

custom_object = WT_NoiseChannelProc.from_tdms(
    tdms_raw_WT[GROUP_NAME][CHAN_NAME],
    desc='Inverter On, WS=0, 100kHz')

plot_spect_comb2([custom_object.calc_spectrum_gen(dec=1, nperseg=100*1024),
                  custom_object.calc_spectrum_gen(dec=10, nperseg=10*1024),
                  custom_object.calc_spectrum_gen(dec=100, nperseg=1024)
                  ],
                 title='Comparison of decimated signal 100kHz',
                 xlim=[1e1, 1e5], ylim=[1e-4, 5e-2]
                 )
```
### filtering a signal with custom_object method `.filter()`
This function is applying a filtering factory method from the package and is 
currently implementing a `butterworth IIR` and a `simple FIR` both low-pass 
filters. For now this is all we need for processing our data (EMI noise after 
5 kHz).

- Default IIR filter with chosen cutoff frequency
``` python
filtered_signal = custom_object.filter(fc_Hz=2000, desc='butterworth low')
```

- Default FIR filter with chosen cutoff frequency
``` python
fc_Hz = 2000
fir_or = 65
# basic low pass FIR filter at 200 Hz 2nd order

fir_filter = fir_factory_constructor(fir_order=fir_or,
                                     fc_Hz=fc_Hz)

filtered_signal = custom_object.filter(fc_Hz=2000, 
                                       desc='simple fir low',
                                       filter_func=fir_filter)
```


### comparing results from `IIR` and `FIR` filters

``` python
from pros_noisefiltering.WT_NoiProc import (filt_butter_factory,
                                            fir_factory_constructor,
                                            plot_comparative_response)

# needs ajustment in some cases 
NPERSEG = 1024
# rectangle plot
FIGSIZE_STD = (6, 6)

# basic low pass butterworth IIR filter at 200 Hz 2nd order
filter_Butter_200 = filt_butter_factory(filt_order=2, fc_Hz=200)

# cutoff frequency and FIR filter order
fc_Hz = 200
fir_or = 65
# basic low pass FIR filter at 200 Hz 2nd order
fir_filter_cnstr_xorder = fir_factory_constructor(fir_order=fir_or,
                                                  fc_Hz=fc_Hz)
FIGSIZE_SQR_L = (8, 10)
plot_comparative_response(custom_object,
                          filter_func=fir_filter_cnstr_xorder,
                          response_offset=2e-4,
                          Kolmogorov_offset=4e0,
                          nperseg=NPERSEG*100,
                          # xlim=0e0,
                          figsize=FIGSIZE_SQR_L)

plot_comparative_response(custom_object,
                          filter_func=filter_Butter_200,
                          response_offset=2e-4,
                          Kolmogorov_offset=4e0,
                          nperseg=NPERSEG*100,
                          figsize=FIGSIZE_SQR_L)

```

### chaining the spectral density (Welch's method) and decimation operations

``` python
NPERSEG=1024<<6
FIGSIZE = (15,10)
plot_spect_comb2([custom_object.decimate(dec=1,offset=0).set_desc('500 kHz').calc_spectrum( nperseg=NPERSEG),
                  custom_object.decimate(dec=10,offset=0).set_desc('50 kHz (dec=10)').calc_spectrum( nperseg=NPERSEG/10),
                  custom_object.decimate(dec=100,offset=0).set_desc('5 kHz (dec=100)').calc_spectrum( nperseg=NPERSEG/100)
                  ],
                title='Comparison of decimated signal 500kHz by two orders of magnitude',
                xlim=[1e1,2.5e5], ylim = [1e-5,0.5e-2],
                figsize = FIGSIZE,
                draw_lines=True
                )
```


