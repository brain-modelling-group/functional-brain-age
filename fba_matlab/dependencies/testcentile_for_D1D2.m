
load fba_fitted_centiles_D1D2 %grab centiles from model
load D1D2_NN_2ch_D1D2.mat % the age and predictor outputs from the D1D2 net/onnx file

% the function does it in a dumb/non optimized way but it work
% here the input is age, predictor per sub (n). the linear offset to adjust age for the older kids, 
% age_centiles are the age bins computed by GAMLSS package in R
% fba_centiles is the 'lookup' table based on the GAMLSS computes based on
% age_centiles and age and predictor of the final NN output
% centiles_tested is the binning of the centiles ranging from 0.5 % to
% 99.5%, with a binning sep of 0.5%
for n=55
%     length(age); 
    centile(n)=fba_centile_estimate(age,pred,n,'offset',offset,age_centiles,fba_centiles,[0.5:0.5:99.5]); 
end


 