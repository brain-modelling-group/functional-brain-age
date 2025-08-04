
% -------------------------------------------
% Functional Brain Age (FBA) Prediction Script
% Using Pretrained Model on Sample EEG Data
% -------------------------------------------

% Load EEG sample and pretrained model
% Grab example file from Zenodo link: https://zenodo.org/records/16734702

load examplefile_eeg_hyp.mat          % Contains 'eeg_ch' [1920 x hyp x epochs], 'hypnogram'; where W = 1, N1 = 2, N2 = 3, N3 = 4, REM = 5; 
load age_sleep_norm_64hz.mat          % Contains pretrained Res-NN model 'net_sleep'; 64 hz version; for EEG resampled to 32 Hz use 32 hz version

% -------------------------------------------
% EEG Assumptions:
% - Filtered with 4th-order Butterworth (0.5–30 Hz bandpass)
% - Resampled to 64 Hz
% - 30-second epochs (1920 samples per epoch)
% Channels: F4-M1, C4-M1, O2-M1, F4-C4, C4-O2
% -------------------------------------------

% Predict FBA from EEG
fba_hypnogram = predict(net_sleep, permute(eeg_ch, [1 3 4 2]));

% state based fba
fba_wake = fba_hypnogram(hypnogram == 1);
fba_n1 = fba_hypnogram(hypnogram == 2);
fba_n2 = fba_hypnogram(hypnogram == 3);
fba_n3 = fba_hypnogram(hypnogram == 4);
fba_rem = fba_hypnogram(hypnogram == 5);

summary = [median(fba_hypnogram) median(fba_wake) median(fba_n1) median(fba_n2) median(fba_n3) median(fba_rem)]

%%

eeg_input = permute(eeg_ch, [1 3 4 2]);  % size: [1920 x 1 x 5 x epochs]
act = activations(net_sleep, eeg_input, 'FC1');