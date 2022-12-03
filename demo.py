#!/usr/bin/env python
"""
Example on how to use FBA functions

@authors: Nathan Stevenson, Kartik Iyer
"""

import argparse

import fba

parser = argparse.ArgumentParser()

parser.add_argument('--edf_filename',
                    default='fba/data/edf/FLE14-609.edf',
                    type=str,
                    help=''' The name of the .edf file with EEG data''')

parser.add_argument('--raw_age',
                    default=0.4,
                    type=float,
                    help='''Raw (Functional Brain Age) or just Age, expressed in years.''')

parser.add_argument('--onnx_filename',
                    default='fba/data/onnx/D1_NN_18ch_model.onnx',
                    type=str,
                    help=''' The name of the .onnx file with the pretrained network.''')

parser.add_argument('--montage_filename',
                    default='fba/data/montages/default_fba_18ch_montage.txt',
                    type=str,
                    help=''' The name of the .txt with the desired EEG montage''')


if __name__ == '__main__':
    args = parser.parse_args()

    # Load, preprocess and chunk EEG data
    eeg_edf = fba.load_edf(args.edf_filename)
    eeg_data = fba.make_montage(eeg_edf, preprocess=True)   # Uses default montage
    eeg_epoch = fba.make_data_epochs(eeg_data)

    # Load pretrained nn model
    onnx_model = fba.onnx_load_model() # Loads default model D1_NN_18ch_model.onnx

    # Estimate FBA
    fba_var = fba.onnx_estimate_fba(onnx_model, eeg_epoch)

    # Estimate centile
    sub_id = 0
    centile, _, _, _ = fba.estimate_centile(args.raw_age, fba_var, sub_id, to_plot=True) # Uses default bins, lookup table and offset

    msg = f"Predicted Functional Brain Age (FBA) is {fba_var} years"
    print(msg)
