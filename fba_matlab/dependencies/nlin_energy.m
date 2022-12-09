%% non-linear enery
function snleo = nlin_energy(dat)
% dat is the eeg data
xm3 = [dat 0 0 0];
xm2 = [0 dat 0 0];
xm1= [0 0 dat 0];
xm0= [0 0 0 dat];
 
nleo = xm1.*xm2 - xm0.*xm3;
nleo = nleo(4:end) ;
sm = 7;% round(0.0675*fs); % vary the smoothing window duration to suit you
dum1 = conv(abs(nleo), ones(1,sm))./sm; % or shape – hamming or hanning instead of ones
snleo = dum1(4:end-3);
snleo = snleo-min(snleo);
end