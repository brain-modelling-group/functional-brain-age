#!/usr/bin/env python
"""

Working example of Functional Brain Age prediction. By default runs with a demo edf and onnx file

@author: Nathan Stevenson, Kartik Iyer
"""

# Base Imports
import os
from pathlib import Path
import argparse

# Scientific Imports
import numpy as np
import scipy.signal as signal
import onnxruntime as rt  # onnx runtime is the bit that deals with the ONNX network


parser = argparse.ArgumentParser()


parser.add_argument('--edf_filename',
                    default='demo-data/random-noise.edf',
                    type=str,
                    help=''' The name of the .edf file with EEG data''')

parser.add_argument('--raw_fba',
                    default=0.4,
                    type=float,
                    help=''' Raw Functional Brain Age, expressed in year.''')

parser.add_argument('--onnx_filename',
                    default='demo-onnx/D1_NN_18ch_model.onnx',
                    type=str,
                    help=''' The name of the .onnx file with the pretrained network.''')

parser.add_argument('--montage_filename',
                    default='demo-data/default_fba_montage.txt',
                    type=str,
                    help=''' The name of the .txt with the desired EEG montage''')


# -- EEG related functions
def load_edf(filename, verbose=True, **kwargs):
    """
    Loads an EDF file. This is a very opinionated edf reader, that loads excatly the data the pretrained onnx model
    needs. Assumes the following properties about the EDF data:
      - encoding is utf-8
      - duration of data records is expressed in seconds

    Parameters:
    -----------
    filename : str | Path
        the filename (or full path) to the file
    folder   : str | Path
        the folder (may not be necesssary now)
    verbose  : bool
        whether to print details
    kwargs   : dict
         passed to maybe some other function

    Returns:
    -------
    edf_info : dict
        a dictionary with lots of information about the EDF we just read

    """
    if isinstance(filename, Path):
        filename = str(filename)

    ext = os.path.splitext(filename)[1][1:].lower()
    if ext in ('edf'):
        try:
            with open(filename, 'rb') as fid:
                edf_info = _read_edf_header(fid)
        except:  # something goes wrong and can't open the file
            pass
    else:
        raise NotImplementedError(
                f'Can only read EDF files, but got a {ext} file.')
    return  edf_info


def _read_edf_header(fid, decode_mode='utf-8'):
    """
    Read the correct bits out of the EDF file

    Parameters:
    -----------
    :param fid:
    :param decode_mode:

    Returns:
    -------

    """

    _   = fid.read(8).decode(decode_mode)   # (Unused) version of this data format (0)
    pid = fid.read(80).decode(decode_mode)  # local patient identification
    rid = fid.read(80).decode(decode_mode)  # local recording identification
    d_start = fid.read(8).decode(decode_mode)  # startdate of recording (dd.mm.yy)
    t_start = fid.read(8).decode(decode_mode)  # starttime of recording (hh.mm.ss)
    # t_start = convert_time(d_start, t_start) convert to seconds from year 2000
    h_len = int(fid.read(8).decode(decode_mode))      # number of bytes in header record
    rsrv1 = fid.read(44)                              # reserved
    n_rec = int(fid.read(8).decode(decode_mode))      # number of data records (-1 if unknown) duration of a data record, in seconds
    if n_rec == -1:
        raise ValueError(
            f'File does not contain eany data records'
        )

    rec_dur = int(fid.read(8).decode(decode_mode))          # duration of a data record, in seconds -- sampling period, in seconds
    nchan = int(fid.read(4).decode(decode_mode))            # number of channels (nchan) in data record
    channels = list(range(nchan))
    chan_labels =  [fid.read(16).strip().decode(decode_mode) for _ in channels]  # nchan * label (e.g. EEG Fpz-Cz or Body temp)
    trnsdcr = [fid.read(80).strip().decode(decode_mode) for _ in channels]       # nchan x transducer type (e.g. AgAgCl electrode)
    phy_dim = [fid.read(8).strip().decode(decode_mode) for _ in channels]        # nchan x physical dimension (e.g. uV or degreeC)
    phy_min = [float(fid.read(8).decode(decode_mode)) for _ in channels]  # nchan x physical minimum (e.g. -500 or 34)
    phy_max = [float(fid.read(8).decode(decode_mode)) for _ in channels]  # nchan x physical maximum (e.g. 500 or 40)
    dig_min =  [int(fid.read(8).decode(decode_mode)) for _ in channels]
    dig_max =  [int(fid.read(8).decode(decode_mode)) for _ in channels]  # nchan * digital maximum (e.g. 2047)
    p_filt = fid.read(nchan * 80).decode(decode_mode)                    # nchan * prefiltering (e.g. HP:0.1Hz LP:75Hz)
    n_samp =  [int(fid.read(8).decode(decode_mode)) for _ in channels]   # nchan * number of samples
    rsrv2 = fid.read(nchan * 32)                                         # reserved
    # Check we haven't messed up in reading the header info
    assert fid.tell() == h_len

    scale  = (np.asarray(phy_max) - np.asarray(phy_min)) / (np.asarray(dig_max) - np.asarray(dig_min))
    offset = (np.asarray(phy_min) + np.asarray(phy_max)) / 2 * (
                   (np.asarray(phy_max) - np.asarray(phy_min)) / (np.asarray(dig_max) - np.asarray(dig_min))
    )

    data = {}
    # setup up data dictionary
    INT16 = np.dtype('<i2')  # (2 Bytes) int 16-bit big endian
    # Read in data in int format
    for kk in range(n_rec):
        dum = np.fromfile(fid, count=sum(n_samp), dtype=INT16)
        if kk == 0:
            r1 = 0
            for jj in range(nchan):
                data[chan_labels[jj]] = dum[r1:r1 + n_samp[jj]]
                # print(np.shape(dum[r1:r1+n_samp[jj]]), n_samp[jj])
                r1 = r1 + n_samp[jj]
        else:
            r1 = 0
            for jj in range(nchan):
                data[chan_labels[jj]] = np.append(data[chan_labels[jj]], dum[r1:r1 + n_samp[jj]])
                r1 = r1 + n_samp[jj]

    edf_info = {'patient_id': pid, 'recording_id': rid, 'start_date': d_start, 'start_time': t_start,
                'num_channels': nchan, 'channel_labels': chan_labels,
                'recording_duration': n_rec * rec_dur, 'transducer_type': trnsdcr,
                'scale': scale, 'offset': offset, 'raw_data': data,
                'channel_units': phy_dim}


    return edf_info


def to_df():
    """
    Save edf properties to a csv file from a pandas dataframe
    should be a method if EDF were a class
    :return:
    """
    pass


def make_montage(montage):
    """
    Either read a montage from a file, or a dictionary with the montage
    Args:
        montage     (str/Path/dict)

    Returns:
    """

    # # this is the montage so I will need to search through to fine the pairs in label, also seems to be in 'Raw Data'
    # # doing this the hard way - ideally you would search through the labels in eeg and assemble that way
    # eeg = edf_info['raw_data']
    # data = np.zeros([225250, 18])
    # data[:, 0] = eeg1['Fp2            '] - eeg1['F4             ']
    # data[:, 1] = eeg1['F4             '] - eeg1['C4             ']
    # data[:, 2] = eeg1['C4             '] - eeg1['P4             ']
    # data[:, 3] = eeg1['P4             '] - eeg1['O2             ']
    # data[:, 4] = eeg1['Fp1            '] - eeg1['F3             ']
    # data[:, 5] = eeg1['F3             '] - eeg1['C3             ']
    # data[:, 6] = eeg1['C3             '] - eeg1['P3             ']
    # data[:, 7] = eeg1['P3             '] - eeg1['O1             ']
    # data[:, 8] = eeg1['Fp2            '] - eeg1['F8             ']
    # data[:, 9] = eeg1['F8             '] - eeg1['T4             ']
    # data[:, 10] = eeg1['T4             '] - eeg1['T6             ']
    # data[:, 11] = eeg1['T6             '] - eeg1['O2             ']
    # data[:, 12] = eeg1['Fp1            '] - eeg1['F7             ']
    # data[:, 13] = eeg1['F7             '] - eeg1['T3             ']
    # data[:, 14] = eeg1['T3             '] - eeg1['T5             ']
    # data[:, 15] = eeg1['T5             '] - eeg1['O1             ']
    # data[:, 16] = eeg1['Fz             '] - eeg1['Cz             ']
    # data[:, 17] = eeg1['Cz             '] - eeg1['Pz             ']
    # data = data * eeg['Scale'][0]
    # data = data[0:15 * 250 * 60, 0:]
    pass


def preprocess(eeg):
    """
    Filter and resample data

    """

    # [b, a] = signal.butter(4, [0.5, 30], btype='bandpass', fs=250)
    # data = signal.filtfilt(b, a, data, axis=0)
    # data = signal.resample(data, 32 * 15 * 60, axis=0)  # so this is a 15 minute segment of EEG
    # # that is filtered and resampled to 32 Hz, the next thing to do is split it up and send it to the NN
    # data.astype('float32')

    pass


def get_data_epoch():
    # n = 29  # number of sample
    # c = 1  # rgb colour depth
    # h = 1920  # epoch length
    # w = 18  # 18 channels
    # aa = np.zeros((n, c, h, w)).astype('float32')
    #
    # for jj in range(0, n):
    #     r1 = jj * 32 * 30
    #     r2 = r1 + 32 * 60
    #     aa[jj, 0, 0:, 0:] = data[r1:r2, :]

    pass


def centiles():
    """
    Kartik's code

    """
    pass


# -- ONNX related functions
def onnx_load_model(filename):
    """ Loads an ONNX file that has been saved from matlab or other

    """

    # onnx_model_file = 'demo-onnx/D1_NN_18ch_model.onnx'  # ANET style network - seznet_v2.onnx is RESNET
    # session = rt.InferenceSession(onnx_model_file)  # load network
    pass


def onnx_estimate_age():
    # input_name = session.get_inputs()[0].name  # find what input layer is called
    #
    # # this is pulling the above data from a mat file into the correct format for ONNX/runtime
    #
    # #
    # # TESTING
    # #
    #
    # result = session.run(None, {
    #     input_name: aa})  # run network - compare this to the outputs variable above to compare ONNX runtime to Matlab
    # outputs1 = result[0]
    # print(np.mean(outputs1))

    pass


# Run as a script
if __name__ == '__main__':

    eeg_edf = load_edf("demo-data/FLE14-609.edf")
    # eed_data = make_montage(eeg_edf, filter=True, resample=True)
    # eeg_epoch = get_data_epoch(eeg_data)
    # onnx_model = onnx_load_model(onnx_model_filename)
    # onnx_estimate_age(onnx_model, eeg_epoch)
