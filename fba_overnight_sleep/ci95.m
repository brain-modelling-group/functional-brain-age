function ci95_interval = ci95(data)
% ci95 - Computes the 95% confidence interval of the mean absolute error 
%        using bootstrapping (1000 samples).
    data = data(:)';
    
    % Compute absolute errors
    abs_errors = abs(data);
    
    % Number of bootstrap samples and data points
    num_bootstrap = 1000;
    num_points = length(abs_errors);
    
    % Preallocate for bootstrapped means
    bootstrap_means = zeros(1, num_bootstrap);
    
    % Bootstrap sampling
    for i = 1:num_bootstrap
        resample_idx = randi(num_points, 1, num_points);
        bootstrap_sample = abs_errors(resample_idx);
        bootstrap_means(i) = nanmean(bootstrap_sample);
    end
    
    % Compute 2.5 and 97.5 percentiles for 95% CI
    ci95_interval = quantile(bootstrap_means, [0.025, 0.975]);
end
