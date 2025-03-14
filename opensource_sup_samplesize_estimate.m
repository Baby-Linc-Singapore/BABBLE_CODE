%% Sample Size Estimation Script
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.

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