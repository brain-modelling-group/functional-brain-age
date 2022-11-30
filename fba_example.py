#!/usr/bin/env python
"""
Working example of Functional Brain Age prediction.
By default runs with a demo edf and onnx file

@author: Nathan Stevenson, Kartik Iyer
"""

# Base Imports
import os
from pathlib import Path

# Scientific Imports
import numpy as np
import scipy.io as io
import scipy.signal as signal
import onnxruntime as rt  # onnx runtime is the bit that deals with the ONNX network

# --------------------------------- EDF related functions -------------------------------------------------------------#
def load_edf(filename, **kwargs):
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

    TODO: this function will be part of an EDF reader/loader class
    """
    if isinstance(filename, Path):
        filename = str(filename)

    ext = os.path.splitext(filename)[1][1:].lower()
    if ext in 'edf':
        try:
            with open(filename, 'rb') as fid:
                edf_info = _read_edf_header(fid)
        except:  # something goes wrong and can't open the file
            pass
    else:
        raise NotImplementedError(
                f'Can only read EDF files, but got a {ext} file.')
    return edf_info


def _read_edf_data(fid, n_rec, n_samp, chan_labels):
    """
    Actually reads the data in the EDF file

    Parameters:
    -----------
    fid : obj
        the object with the open file to read
    n_rec  : float
        number of data records (-1 if unknown) duration of a data record, in seconds
    n_sample  : int
        total number of samples to read = number of channels * number of samples per channel
    chan_labels : list
        list with channel human-readable labels for each channel (e.g. EEG: 'Fpz-Cz' or 'Body temp')

    TODO: this function will be part of an EDF reader/loader class
    """
    data = {
        channel: np.empty((n_rec*n_samp[idx])) for idx, channel in enumerate(chan_labels)
    }
    # setup up data dictionary
    INT16 = np.dtype('<i2')  # (2 Bytes) int 16-bit big endian
    # Read in data in int format
    for kk in range(n_rec):
        dum = np.fromfile(fid, count=sum(n_samp), dtype=INT16)
        start_in = 0
        for jj, channel in enumerate(chan_labels):
            end_in = start_in + n_samp[jj]
            start_out = n_samp[jj]*kk
            end_out = n_samp[jj]*(kk+1)
            data[channel][start_out:end_out] =  dum[start_in:end_in]
            start_in = start_in + n_samp[jj]

    return  data


def _read_edf_header(fid, decode_mode='utf-8'):
    """
    Read the correct bits out of the EDF file

    Parameters:
    -----------
    fid  : obj
           EDF file handle to read
    decode_mode  : str
           specifies the encoding of the EDF header info

    Returns:
    -------
    edf_info  : dict
           a dictionary with all the necessary info about this EDF file, including the data
           in edf_ino['raw_data']

    TODO: this function will be part of an EDF reader/loader class

    """

    _   = fid.read(8).decode(decode_mode)   # (Unused) version of this data format (0)
    pid = fid.read(80).decode(decode_mode)  # local patient identification
    rid = fid.read(80).decode(decode_mode)  # local recording identification
    d_start = fid.read(8).decode(decode_mode)  # startdate of recording (dd.mm.yy)
    t_start = fid.read(8).decode(decode_mode)  # starttime of recording (hh.mm.ss)
    # t_start = convert_time(d_start, t_start) convert to seconds from year 2000
    h_len = int(fid.read(8).decode(decode_mode)) # number of bytes in header record
    _ = fid.read(44)                         # rsrv1 reserved
    n_rec = int(fid.read(8).decode(decode_mode)) # number of data records (-1 if unknown) duration of a data record, in seconds
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
    dig_min = [int(fid.read(8).decode(decode_mode)) for _ in channels]
    dig_max = [int(fid.read(8).decode(decode_mode)) for _ in channels]   # nchan * digital maximum (e.g. 2047)
    _ = fid.read(nchan * 80).decode(decode_mode)                    # p_filt nchan * prefiltering (e.g. HP:0.1Hz LP:75Hz)
    n_samp = [int(fid.read(8).decode(decode_mode)) for _ in channels]   # nchan * number of samples
    _ = fid.read(nchan * 32)                                         # rsrv2 reserved
    # Check we haven't messed up in reading the header info
    assert fid.tell() == h_len

    scale = (np.asarray(phy_max) - np.asarray(phy_min)) / (np.asarray(dig_max) - np.asarray(dig_min))
    offset = (np.asarray(phy_min) + np.asarray(phy_max)) / 2 * (
                   (np.asarray(phy_max) - np.asarray(phy_min)) / (np.asarray(dig_max) - np.asarray(dig_min))
    )

    # Load the data
    data = _read_edf_data(fid, n_rec, n_samp, chan_labels)

    # Dictionary with outputs
    edf_info = {'patient_id': pid,
                'recording_id': rid,
                'start_date': d_start,
                'start_time': t_start,
                'num_channels': nchan,
                'channel_labels': chan_labels,
                'recording_duration': n_rec * rec_dur,
                'transducer_type': trnsdcr,
                'scale': scale,
                'offset': offset,
                'raw_data': data,
                'channel_units': phy_dim}
    return edf_info


def to_df():
    """
    Save edf properties to a csv file from a pandas dataframe
    should be a method if EDF were a class
    Parameters
    ----------

    Returns
    -------
    TODO: this function will be part of an EDF reader/loader class
    """
    pass

# --------------------------------- eeg data  related functions --------------------------------------------------------#

def make_montage(eeg_edf, montage_specs, preprocess=True, **kwargs):
    """
    Either read a montage from a file, or a dictionary with the montage
    Parameters
    ----------
    eeg_data       : dict with edf info and data
    montage_specs  : str or Path or dict
    preprocess     : bool
    kwargs         : dict, optiona
                     passed to _preprocess()

    Returns
    -------
    eeg_data : array
    """

    if isinstance(montage_specs, dict):
        pass
    elif isinstance(montage_specs, (str, Path)):
        pass

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
    if preprocess:
        eeg_data = _preprocess(eeg_data, **kwargs)
    return eeg_data


def _preprocess(eeg_data):
    """
    Filters and then resamples eeg data to 32 Hz, before being chunked and sent to the
    neural network.

    Parameters:
    -----------
    eeg_data : array
        numpy array with eeg data

    Returns:
    -------_
    eeg_data : array
        numpy array with the eeg data filtered and resampled (downsampled)

    TODO: this function will be part of an EDF reader/loader class
    """

    [b, a] = signal.butter(4, [0.5, 30], btype='bandpass', fs=250)
    eeg_data = signal.filtfilt(b, a, eeg_data, axis=0)  # filter
    eeg_data = signal.resample(eeg_data, 32 * 15 * 60, axis=0)  # resample to 32 Hz
    # Change type
    eeg_data.astype('float32')

    return eeg_data


def get_data_epoch(data, n_samples=29, c=1, epoch_length=1920, num_channels=18):
    """
     Extract n_samples data of length epoch_length

    Parameters:
    -----------
    data      : array
                2D array with data to chunk into epochs
    n_samples : int
                number of samples to extract
    c         : int
                rgb colour depth
    epoch length: int
                length of each epock in number of samples
    num_channels : int
                number of channels in the data

    Returns:
    -------
    arr : array
          4D array of shape (n_samples, c, epoch_length, num_channels)

    """
    # Allocate memory
    arr = np.empty((n_samples, c, epoch_length, num_channels)).astype('float32')

    for jj in range(n_samples):
        r1 = jj * 32 * 30
        r2 = r1 + 32 * 60
        arr[jj, 0, ...] = data[r1:r2, ...]

    return arr

# --------------------------------- eeg data  related functions --------------------------------------------------------#


def _load_fitted_centiles(filename=None):
    """
    Load precomputed age and fba centiles, computed via GAMLSS in Rstudio

    Parameters
    ----------
    filename : str | Path
               the filename (or full path) to the file with the stored age and fba centiles
               If filename=None (default) it loads the precomputed data.

    Returns
    -------
    age_centiles    : array
                      an array of shape (n,), where n is the number of subjects in the combined FBA model
                      In the default precomputed data n = 1810.
    centile_centres : array
                      an array of shape (m,), where m is the number of age bins tested.
                      This array contains the bin centres, not the edges.
                      In the default precomputed data n = 199.
                      Bin centres are located at [0.5:0.5:99.5]
    fba_centiles    : array
                      an array of shape (m, n), where n is the number of subjects in the combined FBA model
                                                      m is the number of centiles tested in the combined FBA model

    """
    if filename is not None:
        # NOTE: placeholder in case we want to load something different
        pass

    precomp_centiles = io.loadmat('demo-data/centiles/fba_fitted_centiles_D1D2.mat')

    return precomp_centiles['age_centiles'].flatten(), precomp_centiles['centiles_tested'].flatten(), precomp_centiles['fba_centiles']


def estimate_centile(age_var, fba_var, sub_id, offset_pars, to_plot=False, **kwargs):
    """
     From original in matlab: fba_centile_estimate
     Computes the centile value based on actual empirical age (age_var)
     and functional brain age estimate (fba_var). It utilises
     precomputed centiles spanning the data in the combined FBA
     model (D1&D2)

     Parameters
     ----------
     age_var : float or array
               actual age of the subject / epoch of data. If age_var is an array, then its
               shape is (n, ) where n is the number of subjects.
     fba_var : float or array
               functional brain age estimate obtained with the neural network model.
               If age_var is an array, then its shape is (n, ) where n is the number of subjects.
     sub_id   : int
               subject id
     offset_pars: dict
               various parameters
               offset_pars['offset'] : bool
                                      whether fba_var is adjusted to the linear fit

               offset_pars['values']: float
                                      regression offset for the combined FBA model
    to_plot  : bool
               whether to plot output or not, default=False
    kwargs   : dict, optional
               keyword arguments passed to _load_fitted_centiles() in case you want
               to load a new set of age and fba centiles
    Returns
    -------
    centile : float
              a value between 0 and 100 (%) expressing the centile value
              based on nearest FBA value in fba_centiles

    fig_handle: figure object, optional if to_plot=True

    """
    if offset_pars['offset']:
        # Apply offset
        fba_var = fba_var + (age_var - (offset_pars['value'] * age_var))

    # Get precomputed data
    age_centiles, centiles_tested, fba_centiles = _load_fitted_centiles(**kwargs)

    # NOTE: maybe add a consistency check for dimensions m and n

    # Find the closest centile the actual (empirical) age belongs to
    # [~,age_nearest_idx] = min(bsxfun(@(x,y)abs(x-y),age_var(subid),age_centiles'),[],2);
    age_nearest_idx = np.argmin(np.abs(age_var-age_centiles))

    # Extract the closest fba_centile based on the actual age
    # fba_centiles is of shape (m, n), where m is the number of centiles, and n the number of subjects
    test_fba_centile = fba_centiles[age_nearest_idx, :]

    # [~, fba_nearest_idx] = min(bsxfun( @ (x, y) abs(x - y), fba_var(subid), testcen'),[],2);
    fba_nearest_idx = np.argmin(np.abs(fba_var-test_fba_centile))

    # Extract the closest tested centile based on predicted age (FBA)
    centile = centiles_tested[fba_nearest_idx]

    if to_plot:
        import matplotlib.pyplot as plt

        fig_handle, ax = plt.subplots()

        # Do some plotting
        ax.plot(age_var[sub_id], fba_var[sub_id])
        ax.plot(age_centiles, fba_centiles[fba_nearest_idx, :])

        return centile, fig_handle

    return centile


# -- ONNX related functions
def onnx_load_model(filename=None):
    """
    Loads an ONNX file that has been saved from matlab or other

    Parameters
    ----------
    filename : str | Path
               the filename (or full path) to the file with the stored age and fba centiles
               If filename=None (default) it loads the pre-trained network D1_NN_18ch_model.onnx.

    Returns
    --------
    session :  an ONNX InferenceSession object, the main class of ONNX Runtime. It is used to load and run an ONNX model,
               as well as specify environment and application configuration options.
    """
    if filename is not None:
        onnx_model_file = filename
    else:
        onnx_model_file = 'demo-onnx/D1_NN_18ch_model.onnx'  # ANET style network - seznet_v2.onnx is RESNET

    session = rt.InferenceSession(onnx_model_file)           # load network
    return session


def onnx_estimate_fab(onnx_session, eeg_epochs):
    # this is pulling the above data from a mat file into the correct format for ONNX/runtime
    input_name = onnx_session.get_inputs()[0].name  # find what input layer is called
    #
    result = onnx_session.run(None, {input_name: eeg_epochs})  # run network - compare this to the outputs variable above to compare ONNX runtime to Matlab
    outputs = result[0]
    fab_estimate = np.mean(outputs)
    return fab_estimate


# Run as a script
if __name__ == '__main__':
    import argparse

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

    # This is the sequence of steps
    eeg_edf = load_edf("demo-data/FLE14-609.edf")
    # eed_data = make_montage(eeg_edf, filter=True, resample=True)
    # eeg_epoch = get_data_epoch(eeg_data)

    # onnx_model = onnx_load_model(onnx_model_filename)
    # onnx_estimate_fab(onnx_model, eeg_epoch)
    # use this as centile information?

