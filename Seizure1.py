# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:58:40 2019

@author: Benutzer1
"""
import mne
mne.set_log_level('INFO')
mne.set_config('MNE_LOGGING_LEVEL', 'WARNING', set_env=True)
mne.get_config_path()
import scipy
from scipy import io
import numpy as np
from mne.time_frequency import tfr_morlet, psd_welch
from mne.stats import permutation_t_test
import matplotlib.pyplot as plt
from mne.time_frequency import AverageTFR
#%%Loading Data
raw_fname1 = ('C:/Users/Benutzer1/Desktop/19092019/Seizure1_19channels_final.edf')
raw1 = mne.io.read_raw_edf(raw_fname1)
plot1 = raw1.plot()

ch_names=raw1.ch_names
#ch_names = ('FP1', 'FP2' ,'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1' ,'O2')
sfreq = 256
freqs = np.arange(1., 120, 1.)
n_cycles=freqs/3.
#%%Annotations
info=mne.io.meas_info.create_info(ch_names=ch_names, montage='standard_1020', sfreq=256, ch_types='eeg')
#raw1.info['ch_names']=info['ch_names']
raw1.info['dig']=info['dig']
seizure1_annot = mne.Annotations(onset=375, duration=86, description='seizure1')
raw1.set_annotations(seizure1_annot)

(seizure1_event,
 seizure1_event_dict) = mne.events_from_annotations(raw1, chunk_duration=86)
fig = raw1.plot(start=370, duration=96)
print((seizure1_event[:, 0] - raw1.first_samp) / raw1.info['sfreq'])
seizure1_epoch = mne.Epochs(raw1, seizure1_event, event_id=1, tmin=0, tmax=86)
mne.viz.plot_epochs_image(seizure1_epoch)

#%%
power = tfr_morlet(seizure1_epoch, freqs=freqs,
          n_cycles=n_cycles, use_fft=True,  return_itc=False)
'''
power_list_delta=[]

data = seizure1_epoch.get_data()
n_permutations = 50000
T0, p_values, H0 = permutation_t_test(data, n_permutations, n_jobs=1)

raw1.plot_psd(area_mode='range', tmin=375, tmax=461, show=False, average=True)
seizure1_epoch.plot_psd()
seizure1_epoch.plot_psd_topomap(ch_type='eeg')

AverageTFR(raw1.info, data, times, freqs, nave)
'''
#info=mne.io.meas_info.create_info(ch_names=channels, montage='standard_1020', sfreq=256, ch_types='eeg')
power.plot_topo(baseline=None,fmin=80,fmax=100)
power.plot([3],title=power.ch_names[3],fmin=80,fmax=100)
plt.imshow(power.data[:,0:60,:].mean(axis=0),extent=[0,80,0,60])
plt.colorbar()
