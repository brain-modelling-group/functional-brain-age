function [mae_by_bin, mae_sd, mean_by_bin, sem_by_bin, mae_se, rel_error_by_bin, mae_raw_bins, ci95_by_bin, bin_values] = mae_binned_ci(predictions, targets, bin_edges)

% Calculate residuals and overall MAE
residuals = targets - predictions;
overall_mae = mean(abs(residuals));

% Bin data by prediction age
age_bins = bin_predictions(predictions, bin_edges);

% Initialize outputs
num_bins = length(age_bins.idx);
mae_by_bin = zeros(1, num_bins);
mae_sd = zeros(1, num_bins);
mae_se = zeros(1, num_bins);
mean_by_bin = ones(1, num_bins); % Placeholder
sem_by_bin = ones(1, num_bins);  % Placeholder
rel_error_by_bin = cell(1, num_bins);
ci95_by_bin = zeros(num_bins, 2);
mae_raw_bins = cell(1, num_bins);
bin_values = cell(1, num_bins);

% Loop over bins
for i = 1:num_bins
    idx = age_bins.idx{i};
    res_bin = residuals(idx);
    target_bin = targets(idx);
    
    abs_res_bin = abs(res_bin);
    rel_error = (abs(target_bin - predictions(idx)) ./ predictions(idx)) * 100;

    mae_by_bin(i) = nanmean(abs_res_bin);
    mae_sd(i) = std(abs_res_bin) / sqrt(length(abs_res_bin));
    mae_se(i) = std(abs_res_bin) / sqrt(length(abs_res_bin));
    
    rel_error_by_bin{i} = rel_error;
    mae_raw_bins{i} = res_bin;
    bin_values{i} = target_bin;

    % Bootstrapped 95% CI for MAE
    boot_mae = bootstrp(1000, @(x) mean(abs(x)), res_bin);
    ci95_by_bin(i, :) = quantile(boot_mae, [0.025, 0.975]);
end

end


function age_bins = bin_predictions(ages, bin_edges)
% Bin data using either quantiles or fixed bin edges

if numel(bin_edges) <= 1
    quantiles = quantile(ages, bin_edges);
    bin_edges = [0, quantiles, ceil(max(ages))];
end

for i = 1:length(bin_edges) - 1
    bin_mask = ages > bin_edges(i) & ages <= bin_edges(i + 1);
    age_bins.idx{i} = find(bin_mask);
    age_bins.values{i} = ages(bin_mask);
    age_bins.counts(i) = sum(bin_mask);
end

age_bins.edges = bin_edges;
end
