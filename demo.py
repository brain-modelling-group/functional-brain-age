#!/usr/bin/env python
"""
Example on how to use FBA functions

@authors: Nathan Stevenson, Kartik Iyer
"""

import argparse

import fba

parser = argparse.ArgumentParser()

parser.add_argument('--edf_filename',
                    default='fba/data/edf/random-noise.edf',
                    type=str,
                    help=''' The name of the .edf file with EEG data''')

parser.add_argument('--onnx_filename',
                    default='fba/data/onnx/D1_NN_18ch_model.onnx',
                    type=str,
                    help=''' The name of the .onnx file with the pretrained network.''')

parser.add_argument('--montage_filename',
                    default='fba/data/montages/default_fba_18ch_montage.txt',
                    type=str,
                    help=''' The name of the .txt with the desired EEG montage''')

parser.add_argument('--num_channels',
                    default=18,
                    type=int,
                    help='''Number of EEG channels to pass to the NN model.''')

parser.add_argument('--raw_age',
                    default=0.4,
                    type=float,
                    help='''Raw (Functional Brain Age) or just Age of the data in the EDF file, expressed in years.''')


if __name__ == '__main__':
    args = parser.parse_args()

    # Load, preprocess and chunk EEG data into a shape that the NN uses
    eeg_edf = fba.load_edf(args.edf_filename)
    eeg_data = fba.make_montage(eeg_edf, num_channels=args.num_channels, preprocess=True)   # Uses default 18ch montage
    eeg_epochs = fba.make_data_epochs(eeg_data, num_channels=args.num_channels)

    # Load pretrained nn model
    onnx_model = fba.onnx_load_model(filename=args.onnx_filename) # Loads default model D1_NN_18ch_model.onnx

#------------------------------------------OUTPUTS---------------------------------------------------------------------#

    # Estimated FBA
    fba_var_nn = fba.onnx_estimate_fba(onnx_model, eeg_epochs)

    # Estimate centile
    sub_id = 0
    # Estimate Centile from Growth Chart and potentially correct/offset FBA to align with growth chart.
    centile, fba_var_corrected, _ = fba.estimate_centile(args.raw_age, fba_var_nn, sub_id, to_plot=True) # Uses default bins, lookup table and offset

    msg = f"""
          Empirical/Raw Brain Age is {args.raw_age} years.\n  
          Predicted Functional Brain Age (FBA) is {fba_var_nn} years.\n 
          Estimated centile from Growth Chart is {centile}%.\n
          Corrected Functional Brain Age (FBA) is {fba_var_corrected} years.\n
          Predicted Age Difference (PAD) is: {args.raw_age - fba_var_corrected}"""
    print(msg)
