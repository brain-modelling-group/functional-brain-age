function feat = artefact_features(data, frange, fs)

A = size(data);
val = 11;
feat = zeros(15, 1);
for ii = 1:A(1)
    dat = data(ii,:); 
    env = abs(hilbert(dat));
    feat(1) = mean(env);               % Mean amplitude
    feat(2) = std(env);                    % Variation of amplitude
    d_dat = diff(dat);  
    feat(3) = max(abs(d_dat))/median(abs(dat));                 % Maximum derivative / Mean amplitude
    feat(4) = (skewness(d_dat).^2+1/4*(kurtosis(d_dat)-3)^2);  % JB stat
    [X,f] = pwelch(dat-mean(dat), hamming(2^val), 2^(val-1), [], fs);
    eeg_p = mean(X(find(f*fs>=frange(2,1) & f*fs <= frange(2,2))));
    feat(5) = mean(X(find(f*fs>=frange(1,1) & f*fs < frange(1,2))));     % frequency is too low
    feat(6) = mean(X(find(f*fs>frange(3,1) & f*fs <= frange(3,2))));     % frequency is too high
    feat(7) = eeg_p;  % frequency is just right
    Pxx = X;
    fref = find(Pxx==max(Pxx),1);
    feat(8) = f(fref);
    rr = fref:fref:4*fref; rr = rr(1:find(rr<length(Pxx), 1, 'last'));
    feat(9) = sum(Pxx(rr))/sum(Pxx);
    feat(10) = sum(X-medfilt1(X,11))/sum(X)*100;
    feat(11) = f(find(X==max(X),1));
    feat(12) = sum(f.*X)/sum(X); % Mean Frequency
    feat(13) = sum(f.^2.*X)/sum(X); % Bandwidth
    feat(14) = max(env)/median(env);
    feat(15) = median(env);
end

