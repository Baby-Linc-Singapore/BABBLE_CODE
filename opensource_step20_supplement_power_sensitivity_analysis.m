%% Statistical Sensitivity Summary
% Purpose: Illustrate the MDES-based sensitivity summaries used for the
% main analyses. This script is a reporting template for analysis-level
% estimates and does not rerun raw-data preprocessing.
%
% The focus is on minimum detectable effect size (MDES) benchmarks at the
% target sensitivity level, following the reporting logic described in the
% manuscript and Supplementary Materials.

clear all
clc

fprintf('========================================================================\n');
fprintf('Statistical Sensitivity Summary\n');
fprintf('========================================================================\n\n');

%% Analysis Parameters

n_behavioral = 47;     % Result: behavioral sample reported in manuscript
n_eeg = 42;            % Result: EEG analysis sample reported in manuscript
n_blocks = 226;        % Result: valid block-level observations for GPDC models
alpha_level = 0.05;
target_sensitivity = 0.80;

fprintf('Configuration:\n');
fprintf('  Behavioral sample: %d infants\n', n_behavioral);
fprintf('  EEG sample: %d infants\n', n_eeg);
fprintf('  Block-level GPDC observations: %d\n', n_blocks);
fprintf('  Alpha level: %.2f\n', alpha_level);
fprintf('  Target sensitivity: %.0f%%\n\n', target_sensitivity * 100);

%% MDES Benchmarks

% These values summarize the sensitivity benchmarks reported for the main
% analysis families. Replace values here only if the corresponding
% manuscript-level sensitivity analysis is updated.

analysis_names = {
    'AI alpha GPDC surrogate test';
    'PLS learning prediction';
    'Mediation path a';
    'Mediation path b';
    'Mediation indirect effect'
};

effect_metric = {
    'Cohen d';
    'Cohen f2';
    'beta';
    'beta';
    'beta'
};

mdes_value = [0.52; 0.22; 0.26; 0.19; 0.40];  % Result: reported MDES benchmarks

fprintf('MDES benchmarks at %.0f%% target sensitivity:\n\n', target_sensitivity * 100);
fprintf('%-35s  %-12s  %-12s\n', 'Analysis', 'Metric', 'MDES');
fprintf('%-35s  %-12s  %-12s\n', '---------------------------------', '----------', '----------');

for i = 1:numel(analysis_names)
    fprintf('%-35s  %-12s  >= %.2f\n', analysis_names{i}, effect_metric{i}, mdes_value(i));
end

%% Reporting Summary

fprintf('\nReporting summary:\n');
fprintf('  Sensitivity was summarized using MDES benchmarks for the main\n');
fprintf('  multivariate and mediation analyses. These checks indicate the\n');
fprintf('  approximate effect sizes detectable under the reported sample sizes\n');
fprintf('  and model assumptions.\n\n');

fprintf('========================================================================\n');
fprintf('Script complete.\n');
fprintf('========================================================================\n');
