function feat = single_channel_features(data, frange, fs)
% Function designed to extract
% Time domain, frequency domain, nonlinear/information based measures from
% a single channel of EEG data
%
% Input: data - EEG epoch matrix (M x N) where M is the number of channels
% and N is the length of the epoch in N/fs seconds
%        frange - frequency ranges for spectral power bands of interest
%        fs - sampling frequency of EEG recording
% 
% Output: feat - feature vector that is M x K, where K is the number of
% features calculated on the multi-channel EEG epoch
% feature
% 1-7 various summary statistics of the amplitude envelop defined as the
% magnitude of the analytic associate of the EEG
% 8 - Peak Frequency (PSD(PSD==max(PSD)))
% 9 - power in peak frequency
% 10 - signal amplitude of peak frequency component
% 11 - mean frequency
% 12 - bandwidth
% 13-17 - relative spectral power in bands of interest (set to 5 bands initially)
% 18 - Total power in all bands
% 19 - Spectral slope (decay in power spectrum) 
% 20 - Sample Entropy
% 21 - Fractal Dimension (Higuchi's Method)
% 22 - spectral entropy
% 23 - spectral difference
% 24 - Hjorth 1 
% 25 - Hjorth 2
% 26 - Hjorth 3
% 27 - SNLEO mean 
% 28 - SNLEO std
% 29 - burst shape skewness
% 30 - burst shape kurtosis
% 31 - burst duration (mean)
% 32 - burst duration (standard deviation)
%
% Nathan Stevenson
% QIMRB, 25.05.2021 modified 29/06/2021 with 7 more features


A = size(data); feat = zeros(A(1), 32);
for jj = 1:A(1) % cycle through all channels
    dat = data(jj,:)-mean(data(jj,:));
    % extract some amplitude/time domain features
   % tic
    amp = abs(hilbert(dat));
    amp_m = quantile(amp, [0.05 0.5 0.95]);
    amp_m(4) = mean(amp_m);
    amp_m(5) = std(amp_m);
    amp_m(6) = skewness(amp_m);
    amp_m(7) = kurtosis(amp_m);

    % frequency domain features
    [Pxx, f] = pwelch(dat, hamming(2^10), 2^9, 2^11, fs); % estimate PSD
    fr = find(Pxx==max(Pxx)); 
    fa = f(fr); % Peak Frequency
    rp = sum(Pxx(fr-2:fr+2))/sum(Pxx)*100;
    t = linspace(0,length(dat)/fs,length(dat));
    amp1 = sqrt(mean((sqrt(0.5*Pxx(fr))*sin(2*pi*fa.*t)).^2)); % Amplitude of Peak Frequency 
    mf = sum(f.*Pxx)./sum(Pxx); % Mean frequency
    bw = sum(f.^2.*Pxx)./sum(Pxx); % Bandwidth
    A = size(frange); sp = zeros(1, A(1));
    for ii = 1:A(1)
       fr = find(f>=frange(ii,1) & f<frange(ii,2));
       sp(ii) = sum(Pxx(fr)); % spectral power in defined frequency bands
    end
    rsp = sp./sum(sp)*100; % relative spectral power
    tp = sum(sp); % spectral power in all bands
      
    % information/entropy features
    [Pxx, f] = pwelch(dat, hamming(2^11), 2^10, 2^12, fs);
    fr = 2.^[0.6:0.1:4]-1; P = zeros(1,length(fr)-1);
    for ii = 1:length(fr)-1
       rx = find(f>=fr(ii) & f<(fr(ii+1)));
       P(ii) = mean(Pxx(rx)); 
    end  
   outliers = 1; logf = log2(fr); logP = log2(P); idx = zeros(1,length(P));
   while outliers ~= 0
        x1 = logf(idx==0); y1 = logP(idx==0);
        B = regress(y1', [x1' ones(size(x1'))]);
        res = y1 - (B(1)*x1+B(2)*ones(1,length(x1)));
        [~, idx] = rmoutliers(res);
        outliers = sum(idx);
    end
    a = B(1);  
    dum = conv(dat, ones(1, 8))/8;
    dum = dum(1:8:end);
    msce =  SampEn(2, 0.2*std(dat), dum); % multiscale_entropy(dat); % Estimate MSE and SE  
    [fd,~,~] = Higuchi1Dn(dat); % Estimate Higuchi1Dn
    
    Pxx = Pxx./sum(Pxx);
    se = -sum(Pxx.*log(Pxx+eps))/log(length(Pxx));
    
    N = length(dat); epl = 4*fs; olap = 2*fs; 
    block_no = (N/olap - (epl-olap)/olap); NFFT = 2^ceil(log2(2*epl));
    stft = zeros(block_no, NFFT);
    for kk = 1:block_no
        r1 = (kk-1)*olap+1; r2 = r1+epl-1;
        stft(kk,:) = abs(fft(hamming(epl)'.*(dat(r1:r2)-mean(dat(r1:r2))), NFFT))./epl;
    end
    stft = stft(:, 1:NFFT/2).^2;
    dum = zeros(1, block_no);
    for kk = 1:block_no-1
        dum(kk) = mean((stft(kk,:)-stft(kk+1,:)).^2);
    end
    specd = median(dum, 'omitnan');
        
    activity  = var(dat); % Hjorth 1        
    eeg_diff1 = diff(dat)./fs;   
    mobility = std(eeg_diff1)./activity; % Hjorth 2
    eeg_diff2 = diff(eeg_diff1)./fs; 
    complexity = (std(eeg_diff2)./std(eeg_diff1))./mobility; % Hjorth 3
 
    snleo = nlin_energy(dat);
    sn1 = mean(snleo);
    sn2 = std(snleo);   
    
    %
    dlim = (0.035*fs); % CHANGE TO xlim burst fit, potential limit based on CDF analysis (35ms at the moment - half a fast spike)
    amp = abs(hilbert(dat)).^2;
    amp = conv(amp, [1 1 1 1 1]/5, 'same');
    th1 = quantile(amp, 100); r1 = zeros(1,length(th1));
    for zz = 1:length(th1)
        dum = zeros(1, length(amp));
        dum(amp<th1(zz)) = 0;
        dum(amp>=th1(zz)) = 1;      
        r1x = find(diff([0 dum 0]) == 1);
        r2x = find(diff([0 dum 0]) == -1);
        tst2 = r2x-r1x;
        rf = find(tst2<=dlim);
        for z2 = 1:length(rf)
            dum(r1x(rf(z2)):r2x(rf(z2)))=0;
        end
        r1(zz) = length(find(diff([0 dum 0])==1));
    end
    th = th1(find(r1==max(r1),1)); 
    dum(amp<th) = 0;
    dum(amp>=th) = 1;
    r1x = find(diff([0 dum 0]) == 1);
    r2x = find(diff([0 dum 0]) == -1);
    tst2 = r2x-r1x;
    rf = find(tst2<=dlim);
    for z2 = 1:length(rf)
        dum(r1x(rf(z2)):r2x(rf(z2)))=0;
    end
    
    M = ceil(2*mean(tst2(tst2>dlim))*10);
    bav = zeros(1,M);
    for kk = 1:length(r1)
       if kk>length(r2x); continue; end
       dum =  amp(r1x(kk):r2x(kk)-1)-th; 
       N = length(dum); 
       if N>dlim
           xx = linspace(1,N,M);
           pp = pchip(1:N, dum, xx); 
           pp = pp./sum(pp);
           bav = bav+pp; % Average shape
       end
    end

    % 4 moments of burst shape only need skewness and kurtosis
    bav1 = (bav-min(bav)); bav1 = bav1./sum(bav1);
    xx = linspace(0,1,M);
    mn = sum(xx.*bav1);
    sd = sqrt(sum((xx-mn).^2.*bav1));
    sk = sum((xx-mn).^3.*bav1)./sd^3;
    kt = sum((xx-mn).^4.*bav1)./sd^4-3;
    bdm = mean(tst2(tst2>dlim))/fs;
    bds = std(tst2(tst2>dlim)/fs);
        
    feat(jj,:) = [amp_m, fa, rp, amp1, mf, bw, rsp, tp, a, msce, fd, se, specd, activity, mobility, complexity, sn1, sn2, sk, kt, bdm, bds];
%toc
end
end