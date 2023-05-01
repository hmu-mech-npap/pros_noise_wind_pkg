# npp_wtblade: Processing of Wind turbine data 

This is a package for helping with the processing of Wind Tunnel and Wind Blade measurements. 

Author: Nikolaos Papadakis, N. Torosian

## installation

```console
cd path/to/pkg
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

data = TDMSFile(Path('path/to/data.tdms'))

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
### comparing results from `IIR` and `FIR` filters

``` python
from pros_noisefiltering.WT_NoiProc import (filt_butter_factory,
                                            fir_factory_constructor,
                                            plot_comparative_response)

# needs ajustment in some cases 
NPERSEG = 1024
# rectangle plot
FIGSIZE_STD = (6, 6)

# cutoff frequency and FIR filter order
fc_Hz = 200
fir_or = 65

fir_filter_cnstr_xorder = fir_factory_constructor(fir_order=fir_or,
                                                  fc_Hz=fc_Hz)
FIGSIZE_SQR_L = (8, 10)
plot_comparative_response(df_tdms_1_10,
                          filter_func=fir_filter_cnstr_xorder,
                          response_offset=2e-4,
                          Kolmogorov_offset=4e0,
                          nperseg=NPERSEG*100,
                          # xlim=0e0,
                          figsize=FIGSIZE_SQR_L)

plot_comparative_response(df_tdms_1_10,
                          filter_func=filter_Butter_200,
                          response_offset=2e-4,
                          Kolmogorov_offset=4e0,
                          nperseg=NPERSEG*100,
                          figsize=FIGSIZE_SQR_L)

```

`

