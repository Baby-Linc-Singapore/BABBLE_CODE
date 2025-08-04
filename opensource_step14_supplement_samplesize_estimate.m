%% Sample Size Estimation Script
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Calculate required sample size for detecting a specified effect size
% with desired statistical power in an experimental design
% 
% This script implements an iterative approach to sample size calculation
% based on t-distribution properties rather than using normal approximation

% Input parameters
d = 0.43;      % Target effect size (Cohen's d from previous study)
alpha = 0.05;  % Type I error rate (significance level)
power = 0.80;  % 1 - Type II error rate (statistical power)

% Iterative calculation approach
ncp = 0;       % Non-centrality parameter initialization
n_estimate = 10;  % Initial sample size guess
converged = false;

while ~converged
    % Calculate degrees of freedom based on current sample size estimate
    df = n_estimate - 1;
    
    % Determine critical t-value for the given alpha and df
    t_crit = tinv(1-alpha/2, df);
    
    % Update sample size estimate using the t-distribution properties
    n_new = ceil((t_crit + norminv(power))^2 / d^2);
    
    % Check for convergence (when estimate stabilizes)
    if abs(n_new - n_estimate) < 1
        converged = true;
    else
        n_estimate = n_new;
    end
end

% Display the final required sample size
fprintf('Required sample size: %d\n', n_estimate);
