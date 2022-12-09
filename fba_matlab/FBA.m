function [data,pFBA,cFBA,centile,padAge] = FBA(path,filename,input_age,montage_type,chan_select,mdl_type,mdl)
% INPUTS
%path = folder directory where EEG EDF files are stored
%filename = EEG file of interest
%input_age = age of subject
%montage_type = type of EEG montage (fixed to bipolar montage currently)
%chan_select = empty i.e. [] defaults to all channels, chan_select = 2 selects F4-C4 and C4-O2
%mdl_type = load in relevant pretrained model for GPR or NN
%mdl = depending on chan_select, use pretrained 18ch or 2ch model
%
% OUTPUTS
% data = various EEG montages, referential, bipolar, average montage, and CSD (surface laplacian)
% pFBA = predicted functional brain age 
% cFBA = corrected functional brain age, according to linear offset precomputed from GAMLSS
% centile = centile based on GAMLSS estimates of pretrained model
% padAGE = residual age based on difference between corrected brain age and input_age

%check to add functions and centile dependencies to path
addpath(fullfile(pwd,'dependencies'))
cent=load(fullfile(pwd,'dependencies\fba_fitted_centiles_D1D2.mat')); % computed from GAMLSS

% SELECT MODEL

    
    % LOAD, READ EDF
    [dat, hdr, label, fs, scle, offs] = read_edf(fullfile(path,filename)); % Read in filename
    edf.dat=dat;
    edf.hdr=hdr;
    edf.label=label;
    edf.fs=fs;
    edf.scle=scle;
    edf.offs=offs; %redundant but for initial debugging
    
    %MAKE INTO MONTAGE
    %fs1 = 250; % Re-sampling
    fs1 = fs(1);
    [data_ref_mont, data_bp_mont, data_av_mont, data_csd_mont, stx, rr1] = read_into_montage(edf.dat, edf.label, edf.scle, fs1, edf.fs(1),chan_select);
    
    %EXTRACT SINGLE CHANNEL FEATURES
    frange2 = [0.5 2 ; 2 4 ; 4 8 ; 8 12 ; 12 30];
    epl = 60*fs1; olap = 30*fs1; % Epoch length and Overlap
    [B,A] = butter(4, [1/fs1 60/fs1], 'bandpass'); % bandpass IIR (0.5-30Hz) - 50Hz notch is in read_into_montage function
    data = cell(1,4);
    data{1} = filtfilt(B,A,data_ref_mont')'; %
    data{2} = filtfilt(B,A,data_bp_mont')'; % BIPOLAR Montage;
    data{3} = filtfilt(B,A,data_av_mont')'; %
    data{4} = filtfilt(B,A,data_csd_mont')'; %

if strcmp(mdl_type,'GPR')
    
    bp_channels = stx(:,2);
    if strcmp(montage_type,'bipolar'); %can add more montages if needed
        data1 = data{2};
        block_no = floor(length(data1)/olap)-(epl/olap-1);
        dum_af = cell(1,block_no); dum_sf = dum_af; dum_mf = dum_af;
        for jj = 1:block_no % Epoching - uses epl and olap to define epoch size
            r1 = (jj-1)*olap+1; r2 = r1+epl-1; % Define epoch
            dat1 = data1(:, r1:r2); %dat2 = data2(:, r1:r2); % Select epoch
            dum_sf{jj} = single_channel_features(dat1, frange2, fs1); % estimate single channel features on each channel
            jj
        end
        features_single_channel=dum_sf;
        if chan_select==2
            %for F4-C4 and C4-O2
            eeg_features=features_single_channel;
            cellfind = @(string)(@(cell_contents)(strcmp(string,cell_contents)));
            chan1 = find(cellfun(@(x) ~isempty(strmatch('F4 - C4', x)), bp_channels));
            chan2 = find(cellfun(@(x) ~isempty(strmatch('C4 - O2', x)), bp_channels));
            featav = nanmedian(cat(3,eeg_features{:}),3);
            mfeat = nanmean(featav([chan1 chan2],:),1);
            
        elseif isempty(chan_select)
            %for all channel by default
            eeg_features=features_single_channel;
            featav = nanmedian(cat(3,eeg_features{:}),3);
            mfeat = nanmedian(featav,1);
        end
    end
    
    %PREDICT WITH CV GPR MODEL
    pFBA=predict(mdl,mfeat);
    
    else strcmp(mdl_type,'NN')
    % INSERT NN predict
        load(mdl, 'net')
        fs2 = 32; olap = 30*fs1; epl = 60*fs1;
        data1 = data{2};
        MM = size(data1);
        [MM fs2 fs1]
        block_no = floor(MM(2)/olap)-(epl/olap-1);
        dat1 = single(zeros(fs2*60, MM(1), 1, block_no));
        size(dat1)
        for kk = 1:block_no % Epoching - uses epl and olap to define epoch size
            r1 = (kk-1)*olap+1; r2 = r1+epl-1; % Define epoch
            dat1(:,:,1,kk) = resample(data1(:, r1:r2)', fs2, fs1);        
        end
        age1 = predict(net, dat1);
        pFBA = median(age1);
    
end
    
    %GET CENTILE
    [cFBA,centile] = fba_centile_estimate(input_age,pFBA,1,'offset',cent.offset,cent.age_centiles,cent.fba_centiles,[0.5:0.5:99.5]); 
    padAge=cFBA-input_age;
    
    fprintf(['Empirical Age is ' num2str(input_age)  ' years \n\n'])
    
    fprintf(['Predicted Functional Brain Age (FBA) is ' num2str(pFBA)  ' years \n\n'])
    
    fprintf(['Estimated centile from Growth Chart is ' num2str(centile)  '%% \n\n'])
    
    fprintf(['Corrected Functional Brain Age (FBA) is ' num2str(cFBA)  ' years \n\n'])

    fprintf(['Predicted Age Difference (PAD) is [' num2str(padAge)  '] \n\n'])
    


    function [dat, hdr, label, fs, scle, offs] = read_edf(filename)
        % [data, hdr, label] = read_edf(filename);
        %
        % This functions reads in a EDF file as per the format outlined in
        %  http://www.edfplus.info/specs/edf.html
        %
        % INPUT: filename - EDF file name
        %
        % OUTPUT: dat - a cell array containing the data in the file (int16 format)
        %                   hdr - a cell array the header file information in ASCII format
        %                   label - a cell array containing the labels for each
        %                   channel in dat (ASCII format)
        %
        % Nathan Stevenson
        
        fid = fopen(filename, 'r');
        hdr = cell(1);
        hdr{1} = fread(fid, 256, 'char');         % CONTAINS PATIENT INFORMATION, RECORDING INFORMATION
        %stdt = char(hdr{1}(169:176));
        %stt = char(hdr{1}(177:184));
        %start_time = month_test(str2num(stdt(4:5)'))*24*60*60 + str2num(stdt(1:2)')*24*60*60 + str2num(stt(1:2)')*60*60 + str2num(stt(4:5)')*60+str2num(stt(7:8)');
        len_s = str2num(char(hdr{1}(235:244))');        % START DATE AND TIME and a RESERVED
        %len_s = str2num(len_s)*
        rec_dur = str2num(char(hdr{1}(244:252))');
        ns = char(hdr{1}(253:256))';
        ns = str2num(ns);
        hdr{2} = fread(fid, ns*16, 'char');    % LABEL channel label, temp or HR
        hdr{3} = fread(fid, ns*80,'char');     % TRANSDUCER TYPE
        hdr{4} = fread(fid, ns*8,'char');       % PHYSICAL DIMENSION, voltage - temperature
        hdr{5} = fread(fid, ns*8,'char');       % PHYSICAL MIN
        hdr{6} = fread(fid, ns*8,'char');       % PHYSICAL MAX
        hdr{7} = fread(fid, ns*8,'char');       % DIGITAL MIN
        hdr{8} = fread(fid, ns*8,'char');       % DIGITAL MAX
        label = cell(1);
        for jj=1:ns
            rf2 = jj*8; rf1 = rf2-7;
            label{jj} = char(hdr{2}(16*(jj-1)+1:16*(jj-1)+16));
        end
        try % this is the new bit
            dig_lo = zeros(1,ns); dig_hi = dig_lo; phy_hi = dig_lo; phy_lo = dig_lo;
            for jj = 1:ns
                rf2 = jj*8; rf1 = rf2-7;
                phy_lo(jj) = str2num(char(hdr{5}(rf1:rf2))');
                phy_hi(jj) = str2num(char(hdr{6}(rf1:rf2))');
                dig_lo(jj) = str2num(char(hdr{7}(rf1:rf2))');
                dig_hi(jj) = str2num(char(hdr{8}(rf1:rf2))');
            end
            scle = (phy_hi-phy_lo)./(dig_hi-dig_lo);
            offs = (phy_hi+phy_lo)/2;
        catch
            scle = ones(1,ns)*10000/65536;
            offs = zeros(1,ns);
        end
        
        hdr{9} = fread(fid, ns*80,'char');                    % PRE FILTERING
        hdr{10} = fread(fid, ns*8, 'char');                 % SAMPLING NO rec
        nsamp = str2num(char(hdr{10})');
        hdr{11} = fread(fid, ns*32,'char');     % RESERVED
        fs = nsamp/rec_dur;
        
        % Build the empty data matrix of size INT16 not double
        dat = cell(1, ns);
        for jj = 1:length(nsamp);
            %   len_s.*nsamp(jj)
            dat{jj} = int16(zeros(1,len_s*nsamp(jj)));
        end
        
        % Load data into dat array from EDF file: there are length(nsamp) channels
        % and the size of each channel will be len_s * nsamp(ii)
        for ii = 1:len_s;
            for jj = 1:length(nsamp);
                r1 = nsamp(jj)*(ii-1)+1; r2 = ii*nsamp(jj);
                dat{jj}(r1:r2) = fread(fid, nsamp(jj), 'short')';
            end
        end
        
    end

    function [data_ref_mont, data_bp_mont, data_av_mont, data_csd_mont, stx, rr1] = read_into_montage(dat, label, scle, fs, fs1,chan_select)
        % Takes the data from the Read EDF file and then outputs 3 data types
        % INPUTS
        %   dat - EEG data read in by read_edf - cell array in integer16 format
        %   label - labels of each channel recorded in dat - cell array of
        %   characters
        %   scle - the scaling parameter to convert 16-bit integers into microvolts
        %   fs - sampling frequency of the data (as recorded)
        %   fs1 - sampling frequency of the data (to be processed)
        %   rr1 - is the reference channel, with respect to the referential montage
        %
        % OUTPUTS
        % data_ref_mont = referential montage (reference is on the vertex somewhere Fz,Cz,Pz)
        % data_bp_mont = bipolar montage (double banana)
        % data_av_mont = average montage with a slight twist (corrected for phase lags - sort of)
        % stx - the labels of the various montages (3 x cell array)
        %
        % Nathan Stevenson
        
        str1{1} = 'Fp1'; str1{2} = 'Fp2'; str1{3} = 'F7'; str1{4} = 'F3';
        str1{5} = 'Fz'; str1{6} = 'F4'; str1{7} = 'F8'; str1{8} = 'T3';
        str1{9} = 'C3'; str1{10} = 'Cz'; str1{11} = 'C4'; str1{12} = 'T4';
        str1{13} = 'T5'; str1{14} = 'P3'; str1{15} = 'Pz'; str1{16} = 'P4';
        str1{17} = 'T6'; str1{18} = 'O1'; str1{19} = 'O2';
        ref1 = zeros(1,length(dat));
        for ii = 1:length(dat); ref1(ii) = length(strfind(label{ii}', str1{1})); end
        qq0 = find(ref1==1,1);
        data_ref_mont = zeros(length(str1), length(resample(zeros(1,length(dat{qq0(1)})), fs(qq0(1)), fs1)));
        [B, A] = butter(4, 2*[45 55]/fs1(1), 'stop'); % altered to have wider bandpass
        for jj = 1:length(str1)
            ref1 = zeros(1,length(dat));
            for ii = 1:length(dat)
                ref1(ii) = length(strfind(label{ii}', str1{jj}));
            end
            qq1 = find(ref1==1,1);
            if isempty(dat{qq1})==0
                data_ref_mont(jj,:) = filter(B,A,double(dat{qq1})*scle(qq1));
            end
        end
        A = size(data_ref_mont);
        str = cell(18,2);
        str{1,1} = 'Fp2'; str{1,2} = 'F4';     % Fp2-F4, Fp2-F4
        str{2,1} = 'F4'; str{2,2} = 'C4';     % F4-C4, F4-C4
        str{3,1} = 'C4'; str{3,2} = 'P4';     % C4-P4, C4-P4
        str{4,1} = 'P4'; str{4,2} = 'O2';     % P4-O2, P4-O2
        str{5,1} = 'Fp1'; str{5,2} = 'F3';     % Fp1-F3, Fp1-F3
        str{6,1} = 'F3'; str{6,2} = 'C3';     % F3-C3, F3-C3
        str{7,1} = 'C3'; str{7,2} = 'P3';     % C3-P3, C3-P3
        str{8,1} = 'P3'; str{8,2} = 'O1';     % P3-O1, P3-O1
        str{9,1} = 'Fp2'; str{9,2} = 'F8';     % Fp2-F8, Fp2-F8
        str{10,1} = 'F8'; str{10,2} = 'T4';     % F8-T4, F8-T8
        str{11,1} = 'T4'; str{11,2} = 'T6';     % T4-T6, T8-P8
        str{12,1} = 'T6'; str{12,2} = 'O2';     % T6-O2, P8-O2
        str{13,1} = 'Fp1';  str{13,2} ='F7';     % Fp1-F7, Fp1-F7
        str{14,1} = 'F7'; str{14,2} = 'T3';     % F7-T3, F7-T7
        str{15,1} = 'T3'; str{15,2} = 'T5';     % T3-T5, T7-P7
        str{16,1} = 'T5'; str{16,2} = 'O1';     % T5-O1, P7-O1
        str{17,1} = 'Fz'; str{17,2} = 'Cz';     % Fz-Cz, Fz-Cz
        str{18,1} = 'Cz';  str{18,2} ='Pz';     % Cz-Pz, Cz-Pz
        
        data_bp_mont = zeros(length(str), length(data_ref_mont));
        size(data_bp_mont)
        for jj = 1:length(str)
            ref1 = zeros(1,length(str1));
            ref2 = zeros(1,length(str1));
            for ii = 1:length(str1)
                ref1(ii) = length(strfind(str1{ii}, str{jj,1}));
                ref2(ii) = length(strfind(str1{ii}, str{jj,2}));
            end
            qq1 = find(ref1==1,1);
            qq2 = find(ref2==1,1);
            if length(qq1)==length(qq2)
                data_bp_mont(jj,:) = data_ref_mont(qq1,:)-data_ref_mont(qq2,:);
            end
        end
        
        rr = sum(abs(data_ref_mont),2);
        rr1 = find(rr==min(rr));
        rr2 = find(rr~=min(rr));
        mref = hilbert(mean(data_ref_mont(rr2,:))); % this assumption is that the average here is an estimate of ref
        mref = mref./norm(mref);
        data_av_mont = zeros(A);
        for ii = 1:A(1)
            w = (data_ref_mont(ii,:)*(mref'));
            data_av_mont(ii,:) = data_ref_mont(ii,:)-2*real(mref'*w)';
        end
        
        str2 = str1; % convert to EGI Geodesics 129 format if necessary
        str2{8} = '46'; str2{12} = '109'; str2{13} = '58'; str2{17} = '97';
        M = ExtractMontage('10-5-System_Mastoids_EGI129.csd',str2');
        [G,H] = GetGH(M);
        data_csd_mont = CSD([data_ref_mont(1:rr1-1,:) ; zeros(1,length(data_ref_mont)) ; data_ref_mont(rr1+1:end,:)], G, H);
        
        stx = cell(A(1), 4);
        for ii = 1:A(1); stx{ii,1} = [str1{ii} ' - REF']; stx{ii,3} = [str1{ii} ' - Av']; stx{ii,4} = str1{ii}; end
        if chan_select==2
            str{19,1} = 'C4';  str{19,2} ='O2';     % C4-O2
            for ii = 1:A(1); stx{ii,2} = [str{ii,1} ' - ' str{ii,2}]; end
        else
            for ii = 1:A(1)-1; stx{ii,2} = [str{ii,1} ' - ' str{ii,2}]; end
        end
        
        
        
    end

    function [fba_corr,centile]=fba_centile_estimate(age_var,fba_var,subid,adjust,offset_param,age_centiles,fba_centiles,centiles_tested)
        % fba_centile_estimate computes the centile value based on actual age and functional brain age estimate
        % and utilises precomputed centiles spanning the data in the combined FBA
        % model (D1&D2)
        
        %inputs
        %age_var = actual age
        %fba_var = functional brain age
        %subid = subject id
        %adjust = uses choose whether fba is adjusted to the linear fit;
        %offset_param = regression offset for the combined FBA model
        %age_centiles = age centiles; computed via GAMLSS in Rstudio
        %fba_centiles = fba centiles; computed via GAMLSS in Rstudio.
        %This is a m x n matrix where m is the number of centiles tested (see
        %below) and n is the number of subjects in the combined FBA model
        %centiles_tested = centile binning set to [0.5:0.5:99.5]
        
        %output
        %centile = centile value based on nearest FBA value in fba_centiles
        
        % linear offset for projecting onto final chart
        if strcmp(adjust,'offset')
            fba_corr = fba_var+(age_var-((offset_param(2)*age_var)));
        elseif strcmp(adjust,'none')
            %no adjustment of fba
        end
        
        [~,age_nearest_idx] = min(bsxfun(@(x,y)abs(x-y),age_var(subid),age_centiles'),[],2);
        testcen=fba_centiles(:,age_nearest_idx);
        
        [~,fba_nearest_idx] = min(bsxfun(@(x,y)abs(x-y),fba_corr(subid),testcen'),[],2);
        centile=(centiles_tested(fba_nearest_idx));
        
        % Test figure, uncomment to see centile fit
        figure, plot(age_var,fba_corr,'ko')
        hold on, plot(age_var(subid),fba_corr(subid),'go','MarkerFaceColor','g','MarkerSize',10)
        hold on, plot(age_centiles,fba_centiles(fba_nearest_idx,:),'b','LineWidth',2)
        
    end


end



