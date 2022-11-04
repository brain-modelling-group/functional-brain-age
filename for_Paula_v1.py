# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:21:00 2022

@author: nstevenson
"""


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import onnxruntime as rt # onnx runtime is the bit that deals with the ONNX network
# need to install onnxruntime

#eeg = {} # use dictionaries here so function output will be eeg

#
#
#  this is the read EDF part for the demo file
#
#
    
fid = open("C:\\Users\\nstevenson\\Desktop\\Python\\FBA\\FLE14-609.edf", "rb")
# the age of this EEG is 0.4 years

ver = fid.read(8)
ver = ver.decode("utf-8") # version of this data format (0)
pid = fid.read(80)
pid = pid.decode("utf-8") # local patient identification
rid = fid.read(80)
rid = rid.decode("utf-8") # local recording identification
d_start = fid.read(8)
d_start = d_start.decode("utf-8") # startdate of recording (dd.mm.yy)
t_start = fid.read(8)
t_start = t_start.decode("utf-8") # starttime of recording (hh.mm.ss)
#t_start = convert_time(d_start, t_start) convert to seconds from year 2000
h_len = fid.read(8)
h_len = int(h_len.decode("utf-8")) # number of bytes in header record
rsrv1 = fid.read(44) # reserved
n_rec = fid.read(8)
n_rec = int(n_rec.decode("utf-8")) # number of data records (-1 if unknown) duration of a data record, in seconds
rec_dur = fid.read(8)
rec_dur = int(rec_dur.decode("utf-8")) # duration of a data record, in seconds
n_sig = fid.read(4)
n_sig = int(n_sig.decode("utf-8")) # number of signals (ns) in data record
label = fid.read(n_sig*16)
label = label.decode("utf-8") # ns * label (e.g. EEG Fpz-Cz or Body temp) 
trnsdcr = fid.read(n_sig*80)
trnsdcr = trnsdcr.decode("utf-8") # ns * transducer type (e.g. AgAgCl electrode)
phy_dim = fid.read(n_sig*8)
phy_dim = phy_dim.decode("utf-8") # ns * physical dimension (e.g. uV or degreeC)
phy_min = []
for ii in range(n_sig):
    dum = fid.read(8)
    dum = float(dum.decode("utf-8")) # ns * physical minimum (e.g. -500 or 34)
    phy_min.append(dum)
phy_max = []
for ii in range(n_sig):
    dum = fid.read(8)
    dum = float(dum.decode("utf-8")) # ns * physical maximum (e.g. 500 or 40)
    phy_max.append(dum)
dig_min = []
for ii in range(n_sig):
    dum = fid.read(8)
    dum = int(dum.decode("utf-8")) # ns * digital minimum (e.g. -2048)
    dig_min.append(dum)
dig_max = []
for ii in range(n_sig):
    dum = fid.read(8)
    dum = int(dum.decode("utf-8")) # ns * digital maximum (e.g. 2047)
    dig_max.append(dum)
p_filt = fid.read(n_sig*80)
p_filt = p_filt.decode("utf-8") # ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)
n_samp = []
for ii in range(n_sig):
    dum = fid.read(8)
    dum = int(dum.decode("utf-8")) # ns * digital maximum (e.g. 2047)
    n_samp.append(dum)

rsrv2 = fid.read(n_sig*32) # reserved

data = {}
# setup up data dictionary
dt = np.dtype('<i2') # (2 Bytes) 16-bit big endian 
# Read in data in int format
for kk in range(n_rec):
    dum = np.fromfile(fid, count=sum(n_samp), dtype = dt) 
    if kk == 0:
        r1 = 0
        for jj in range(n_sig):
            data[label[jj*16:(jj+1)*16-1]] = dum[r1:r1+n_samp[jj]]
            #print(np.shape(dum[r1:r1+n_samp[jj]]), n_samp[jj])
            r1 = r1+n_samp[jj]     
    else:
        r1 = 0    
        for jj in range(n_sig):
            data[label[jj*16:(jj+1)*16-1]] = np.append(data[label[jj*16:(jj+1)*16-1]],dum[r1:r1+n_samp[jj]])
            r1 = r1+n_samp[jj]

fid.close()

# Save it all up as a dicitonary, though I don't know if this is the best way to do it

eeg = {'Patient ID':pid, 'Recording ID':rid, 'Start Date':d_start, 'Start Time':t_start, 'Channel Number':n_sig, 'Recording Duration': n_rec*rec_dur, 'Tranducer Type':trnsdcr, 'Scale':(np.asarray(phy_max)-np.asarray(phy_min))/(np.asarray(dig_max)-np.asarray(dig_min)), 'Offset':(np.asarray(phy_min)+np.asarray(phy_max))/2*((np.asarray(phy_max)-np.asarray(phy_min))/(np.asarray(dig_max)-np.asarray(dig_min))), 'Raw Data': data}


[b,a] = signal.butter(4, [0.5, 30], btype='bandpass', fs = 250)
# this is the montage so I will need to search through to fine the pairs in label, also seems to be in 'Raw Data'
# doing this the hard way - ideally you would search through the labels in eeg and assemble that way
eeg1 = eeg['Raw Data']
data = np.zeros([225250,18])
data[:,0] = eeg1['Fp2            ']-eeg1['F4             ']
data[:,1] = eeg1['F4             ']-eeg1['C4             ']
data[:,2] = eeg1['C4             ']-eeg1['P4             ']
data[:,3] = eeg1['P4             ']-eeg1['O2             ']
data[:,4] = eeg1['Fp1            ']-eeg1['F3             ']
data[:,5] = eeg1['F3             ']-eeg1['C3             ']
data[:,6] = eeg1['C3             ']-eeg1['P3             ']
data[:,7] = eeg1['P3             ']-eeg1['O1             ']
data[:,8] = eeg1['Fp2            ']-eeg1['F8             ']
data[:,9] = eeg1['F8             ']-eeg1['T4             ']
data[:,10] = eeg1['T4             ']-eeg1['T6             ']
data[:,11] = eeg1['T6             ']-eeg1['O2             ']
data[:,12] = eeg1['Fp1            ']-eeg1['F7             ']
data[:,13] = eeg1['F7             ']-eeg1['T3             ']
data[:,14] = eeg1['T3             ']-eeg1['T5             ']
data[:,15] = eeg1['T5             ']-eeg1['O1             ']
data[:,16] = eeg1['Fz             ']-eeg1['Cz             ']
data[:,17] = eeg1['Cz             ']-eeg1['Pz             ']
data = data*eeg['Scale'][0]
data = data[0:15*250*60,0:]
data = signal.filtfilt(b,a, data, axis=0)
data = signal.resample(data, 32*15*60, axis=0)  # so this is a 15 minute segment of EEG
# that is filtered and resampled to 32 Hz, the next thing to do is split it up and send it to the NN
data.astype('float32')


onnx_model_file = 'D1_NN_18ch_model.onnx'                   # ANET style network - seznet_v2.onnx is RESNET
session = rt.InferenceSession(onnx_model_file)       # load network
input_name = session.get_inputs()[0].name            # find what input layer is called

# this is pulling the above data from a mat file into the correct format for ONNX/runtime

#
# TESTING
#

n = 29 # number of sample
c = 1 # rgb colour depth
h = 1920 # epoch length
w = 18 # 18 channels
aa = np.zeros((n,c,h,w)).astype('float32')

for jj in range(0,n):
    r1 = jj*32*30 
    r2 = r1+32*60
    aa[jj,0,0:,0:] = data[r1:r2, :]
       
result = session.run(None, {input_name: aa})  # run network - compare this to the outputs variable above to compare ONNX runtime to Matlab
outputs1 = result[0]
print(np.mean(outputs1))
