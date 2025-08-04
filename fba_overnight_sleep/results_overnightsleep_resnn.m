% To run below, download from Zenodo link: https://zenodo.org/records/16734702

% Neural Network readouts to process
model_files = {'results_resnn_hypnogram_ovn_net_v1.mat','results_resnnseq_hypnogram_ovn_net_v1.mat'};

% Load hyp from resnn file for reuse as they are the same for resnnseq
load('results_resnn_hypnogram_ovn_net_v1.mat', 'hyp');
hyp_resnn = hyp;  

tabl1_all = [];  % To collect final results table

for model_idx = 1
    % :length(model_files)
    load(model_files{model_idx});
    
    % For resnnseq, override hyp with hyp from resnn
    if contains(model_files{model_idx}, 'resnnseq')
        hyp = hyp_resnn;
    end

    % Aggregate subject-level FBA and age data
    
    fba_all = [];
    age_all = [];
    mae_folds = zeros(length(aget), 1);

    for fold = 1:length(aget)
        test_ids = idt{fold};      % subject IDs for this fold
        fba_preds = nn_out{fold};  % predicted FBA
        ages = aget{fold};         % actual ages

        unique_subjects = unique(test_ids);
        num_subjects = length(unique_subjects);

        fba_per_subject = zeros(1, num_subjects);
        age_per_subject = zeros(1, num_subjects);

        for s = 1:num_subjects
            subject_idx = find(test_ids == unique_subjects(s));
            fba_per_subject(s) = median(fba_preds(subject_idx));
            age_per_subject(s) = mode(ages(subject_idx));
        end

        mae_folds(fold) = mean(abs(fba_per_subject - age_per_subject));

        fba_all = [fba_all, fba_per_subject];
        age_all = [age_all, age_per_subject];
    end

    % Per-fold hypnogram state-based FBA calculations
    
    f_w = []; f_n1 = []; f_n2 = []; f_n3 = []; f_rem = [];
    fb_1st = []; fb_2nd = []; fb_3rd = [];

    for k = 1:length(idt)
        subject_ids = unique(idt{k});
        num_subjects = length(subject_ids);

        sum_w = zeros(1, num_subjects);
        sum_n1 = zeros(1, num_subjects);
        sum_n2 = zeros(1, num_subjects);
        sum_n3 = zeros(1, num_subjects);
        sum_rem = zeros(1, num_subjects);

        fba_w = nan(1, num_subjects);
        fba_n1 = nan(1, num_subjects);
        fba_n2 = nan(1, num_subjects);
        fba_n3 = nan(1, num_subjects);
        fba_rem = nan(1, num_subjects);
        age_s = nan(1, num_subjects);

        fba_1st = nan(1, num_subjects);
        fba_2nd = nan(1, num_subjects);
        fba_3rd = nan(1, num_subjects);

        for nn = 1:num_subjects
            subj_idx = (idt{k} == subject_ids(nn));
            age_hyp = aget{k}(subj_idx);
            fba_hyp = nn_out{k}(subj_idx);
            hyp_subj = hyp{k}(subj_idx);  % Use overridden hyp for resnnseq

            fba_w(nn)   = nanmedian(fba_hyp(hyp_subj == 1));
            fba_n1(nn)  = nanmedian(fba_hyp(hyp_subj == 2));
            fba_n2(nn)  = nanmedian(fba_hyp(hyp_subj == 3));
            fba_n3(nn)  = nanmedian(fba_hyp(hyp_subj == 4));
            fba_rem(nn) = nanmedian(fba_hyp(hyp_subj == 5));

            sum_w(nn)   = sum(hyp_subj == 1);
            sum_n1(nn)  = sum(hyp_subj == 2);
            sum_n2(nn)  = sum(hyp_subj == 3);
            sum_n3(nn)  = sum(hyp_subj == 4);
            sum_rem(nn) = sum(hyp_subj == 5);

            age_s(nn) = mode(age_hyp);

            n_total = length(fba_hyp);
            i1 = round(n_total / 3);
            i2 = round(2 * n_total / 3);

            fba_1st(nn) = median(fba_hyp(1:i1));
            fba_2nd(nn) = median(fba_hyp(i1+1:i2));
            fba_3rd(nn) = median(fba_hyp(i2+1:end));
        end

        fbaw{k}   = fba_w;
        fban1{k}  = fba_n1;
        fban2{k}  = fba_n2;
        fban3{k}  = fba_n3;
        fbarem{k} = fba_rem;

        fb_1st = [fb_1st, fba_1st];
        fb_2nd = [fb_2nd, fba_2nd];
        fb_3rd = [fb_3rd, fba_3rd];
    end

    % summary performance metrics
    age_bins = 0:2:19;

    % Overall MAE and CI
    mae_total = mean(abs(fba_all - age_all));
    civ = ci95(fba_all - age_all);

    % Binned metrics
    [mae_by_bin, mae_sd, mean_by_bin, sem_by_bin, mae_se, rel_error_by_bin, mae_raw_bins, ci95_by_bin, bin_values] = ...
        mae_binned_ci(age_all, fba_all, age_bins);
    wmae = mean(mae_by_bin);

    % Store resnn summary row or resnnseq summary row in tabl1_all
    tabl1_summary = [mae_total, civ, wmae, mean(ci95_by_bin, 1)];

    % Sleep-stage specific calculations
    sleep_stages = {'W', 'N1', 'N2', 'N3', 'REM'};
    fba_data = {horzcat(fbaw{:}), horzcat(fban1{:}), horzcat(fban2{:}), horzcat(fban3{:}), horzcat(fbarem{:})};

    summary_table = zeros(numel(sleep_stages), 6); % Columns: MAE, CI_low, CI_high, wMAE, mean_bin_CI_low, mean_bin_CI_high

    for ii = 1:numel(sleep_stages)
        stage = sleep_stages{ii};
        data = fba_data{ii};

        mae_stage = nanmean(abs(data - age_all));
        ci_stage = ci95(data - age_all);

        [mae_by_bin_stage, ~, ~, ~, ~, ~, ~, ci95_by_bin_stage, ~] = mae_binned_ci(age_all, data, age_bins);
        wmae_stage = nanmean(mae_by_bin_stage);
        mean_bin_ci_stage = nanmean(ci95_by_bin_stage, 1);

        summary_table(ii, :) = [mae_stage, ci_stage, wmae_stage, mean_bin_ci_stage];
    end

    % Append the overall and sleep stage summary to final table
    tabl1_all = [tabl1_all; round([tabl1_summary; summary_table], 3, 'significant')];
end

%%

% Display the combined table (Table 1 of manuscript)
% First 6 rows = resnn, next 6 rows = resnnseq
disp('Summary table: rows 1-6 = ResNN; rows 7-12 = ResNNSeq')
disp(tabl1_all)
