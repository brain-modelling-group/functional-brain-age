#!/usr/bin/env python
"""
Functions related to Functional Brain Age prediction.

@authors: Nathan Stevenson, Kartik Iyer
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
         passed to maybe some other function (not used at the moment)

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
            # TODO: actually handle the exception here, bare exceptions are bad
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
            data[channel][start_out:end_out] = dum[start_in:end_in]
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


def to_dataframe(filename=None, save=False):
    """
    Save edf properties to  a pandas dataframe, and optionally saves
    it as a csv file. Should be a static method if EDF were a class
    Parameters
    ----------

    Returns
    -------
    TODO: this function will be part of an EDF reader/loader class
    """
    pass


# --------------------------------- eeg data  related functions -------------------------------------------------------#
def _load_montage(filename=None):
    """

    Parameters
    ----------
    filename  : str or Path
        name of file, or path to file with the desired montage, specified as:
        Fp1-F7
        F7-T3
        T3-T5

    Returns
    -------
        montage_specs  : dict
            a dictionary with key value pairs, where key is a "channel a" and value is "channel b"
            such that the resulting eeg data will be returned will be channel_a - channel_b

    #TODO: generalise to accommodate unipolar montages
    """

    if filename is None:
        filename = 'demo-data/montages/demo_montage.txt'

    montage_specs = dict()
    with open(filename, 'r+') as montage:
        channels = montage.read().split('\n')
        for channel in channels:
            ch_a = channel.split('-')[0]
            ch_b = channel.split('-')[1]
            montage_specs[ch_a] = ch_b
    return montage_specs


def make_montage(edf_eeg, montage_specs=None, preprocess=True):
    """
    Either read a montage from a file, or a dictionary with the montage
    Parameters
    ----------
    edf_eeg : dict
        dictionary with all the edf info and data
    montage_specs  : str or Path or dict

    preprocess     : bool
         whether to apply preprocessing or not, if True (default) then the data is filtered and downsampled to 32Hz.

    Returns
    -------
    eeg_data : array
        numpy array with the eeg_data to be passed to the make_data_epochs()
    """
    # Check is montage is specified in a text file

    if montage_specs is None:
       montage_specs = _load_montage()
    elif isinstance(montage_specs, (str, Path)):
        filename = montage_specs
        montage_specs: dict = _load_montage(filename)

    eeg_data = np.zeros_like(edf_eeg['raw_data'])

    for (ch_a, ch_b), idx in enumerate(zip(montage_specs.keys(), montage_specs.values())):
        eeg_data[..., idx] = edf_eeg['raw_data'][ch_a] - edf_eeg['raw_data'][ch_b]
    eeg_data = eeg_data * edf_eeg['scale'][0]
    eeg_data = eeg_data[0:15 * 250 * 60, 0:]

    if preprocess:
        eeg_data = _preprocess(eeg_data)
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


def make_data_epochs(data, n_samples=29, c=1, epoch_length=1920, num_channels=18):
    """
    Extracts n_samples data of length epoch_length

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

# --------------------------------- eeg data  related functions -------------------------------------------------------#


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
                      an array of shape (n, m), n is the number of subjects in the combined FBA model
                                                m is the number of centiles tested in the combined FBA model

    """
    if filename is None:
        filename = 'demo-data/centiles/fba_fitted_centiles_D1D2.mat'
    precomp_centiles = io.loadmat(filename)
    offset_pars = dict()
    offset_pars['offset'] = True
    offset_pars['value'] = precomp_centiles['offset'][1] # Take second value of precomputed values

    return precomp_centiles['age_centiles'].flatten(), precomp_centiles['fba_centiles'], offset_pars


def estimate_centile(age_var, fba_var, sub_id, centile_bin_centres=np.r_[0.5:100:0.5], offset_pars=None, to_plot=False, **kwargs):
    """
     From original in matlab: fba_centile_estimate.m
     Computes the centile value based on actual empirical age (age_var)
     and functional brain age estimate (fba_var). It utilises
     precomputed centiles spanning the data in the combined FBA
     model (D1&D2)

     Parameters
     ----------
     age_var : float or array
               actual age of the subject / epoch of data. If age_var is an array, then its
               shape is (n, ) where n is the number of subjects. Must be a column vector.
               If the array has more axes, or is a row vector, the code will not work, or provide
               wrong results.
     fba_var : float or array
               functional brain age prediction obtained with the neural network model.
               If fba_var is an array, then its shape is (n, ) where n is the number of subjects.
               Must be column vector.
     sub_id   : int
               subject id
     offset_pars: dict
               offset_pars['offset'] : bool
               whether fba_var is adjusted to the linear fit
               offset_pars['values']: float
               regression offset for the combined FBA model
    centile_bin_centres : array
               centres of centile bins that were tested in RStudio using the GAMLSS package
               There are m
    to_plot  : bool
               whether to plot output or not, default=False
    kwargs   : dict, optional
               keyword arguments passed to _load_fitted_centiles() in case you want
               to load a new set of age and fba centiles
    Returns
    -------
    centile : float or array
              a value between 0 and 100 (%) expressing the centile value
              based on nearest FBA value in fba_centiles

    fig_handle: figure object, optional if to_plot=True

    """
    # Get precomputed data

    if offset_pars is None:
        age_centiles, fba_centiles, offset_pars = _load_fitted_centiles(**kwargs)
    else:
        age_centiles, fba_centiles, _ = _load_fitted_centiles(**kwargs)

    num_centiles = centile_bin_centres.shape[0]
    num_subjects = age_var.shape[0]
    if offset_pars['offset']:
        # Apply offset
        fba_var = fba_var + (age_var - (offset_pars['value'] * age_var))

    # NOTE: maybe add a consistency check for dimensions m and n
    # Get indices of subjects
    if not isinstance(age_var, np.ndarray):  # Assume it's a scalar then
        # Find the closest centile the actual (empirical) age belongs to
        # [~,age_nearest_idx] = min(bsxfun(@(x,y)abs(x-y),age_var(subid),age_centiles'),[],2);
        age_nearest_idx = np.argmin(np.abs(age_var - age_centiles))
        age = age_var
    else:
        age_var = age_var[:, np.newaxis]
        age_nearest_idx = np.argmin(np.abs(np.tile(age_var, (1, num_subjects)) - age_centiles), axis=1)
        age = age_var[sub_id]

    # Extract the closest fba_centile based on the actual age
    # fba_centiles is of shape (m, n), where n the number of subjects, where m is the number of centiles
    # test_fba_centile is of shape (n, m), where n the number of subjects, where m is the number of centiles
    test_fba_centile = fba_centiles[:, age_nearest_idx].T

    # Check dimensions are ok
    assert test_fba_centile.shape == (num_subjects, num_centiles)

    if not isinstance(age_var, np.ndarray):
        # [~, fba_nearest_idx] = min(bsxfun( @ (x, y) abs(x - y), fba_var(subid), testcen'),[],2);
        fba_nearest_idx = np.argmin(np.abs(fba_var - test_fba_centile))
        # Vars for plotting
        fba = fba_var
        fba_idx = fba_nearest_idx
    else:
        fba_var = fba_var[:, np.newaxis]
        fba_nearest_idx = np.argmin(np.abs(np.tile(fba_var, (1, num_centiles)) - test_fba_centile), axis=1)
        # Vars for plotting
        fba = fba_var[sub_id]
        fba_idx = fba_nearest_idx[sub_id]

    # Extract the closest tested centile based on predicted age (FBA)
    centile = centile_bin_centres[fba_nearest_idx]

    if to_plot:
        import matplotlib.pyplot as plt
        fig_handle, ax = plt.subplots()

        # Plot all points in age and fba
        ax.plot(age_var, fba_var, color=[0.55, 0.55, 0.55], marker='o', linestyle='none', markersize=4)
        # Plot centiles as thin lines
        ax.plot(age_centiles, fba_centiles.T, color=[0.0, 0.0, 0.0, 0.05])
        # Plot centile of sub_id
        ax.plot(age_centiles, fba_centiles[fba_idx, :].T, color='blue', linewidth=2)
        # Plot sub_id single subject
        ax.plot(age, fba, color='green', marker='o', linestyle='none', markersize=12)
        ax.set_xlim([0, 18])
        ax.set_ylim([0, 20])
        ax.set_xlabel('age [years]')
        ax.set_ylabel('FBA [years]')
        plt.show()

        return centile, age_nearest_idx, fba_nearest_idx, fig_handle

    return centile, age_nearest_idx, fba_nearest_idx


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
    session :  an ONNX InferenceSession object, main class of ONNX Runtime. It is used to load and run an ONNX model,
               as well as specify environment and application configuration options.
    """
    if filename is not None:
        onnx_model_file = filename
    else:
        onnx_model_file = 'demo-onnx/D1_NN_18ch_model.onnx'  # ANET style network - seznet_v2.onnx is RESNET

    session = rt.InferenceSession(onnx_model_file)           # load network
    return session


def onnx_estimate_fba(onnx_session, eeg_epochs):
    # this is pulling the above data from a mat file into the correct format for ONNX/runtime
    input_name = onnx_session.get_inputs()[0].name  # find what input layer is called
    result = onnx_session.run(None, {input_name: eeg_epochs})  # run network - compare this to the outputs variable
                                                               # aboveto compare ONNX runtime to Matlab
    outputs = result[0]
    fba_estimate = np.mean(outputs)
    return fba_estimate

