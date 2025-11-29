%% Identify Significant Connections via Surrogate Test
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to
% NTU's open access policy.
%
% Purpose: Identify connections significantly stronger than surrogate baseline
%
% IMPORTANT: Non-Circular Feature Selection
% This step identifies significant connections by comparing real GPDC values
% against surrogate (phase-randomized) GPDC values. The selection is based
% SOLELY on whether connectivity is stronger than chance, WITHOUT using
% learning outcome data. This ensures the feature selection is independent
% of the outcome variable, preventing circular analysis.
%
% Statistical approach:
% 1. For each connection, compute mean GPDC across participants
% 2. Compare against distribution of surrogate mean GPDC values
% 3. P-value = proportion of surrogates exceeding real data
% 4. Apply FDR correction for multiple comparisons
%
% This approach follows recommendations from:
% - Ding et al. (2006). Granger causality: Basic theory and application to neuroscience
% - Bastos & Schoffelen (2016). A tutorial review of functional connectivity analysis methods
%
% Based on: scripts_R1/fs5_strongpdc.m

clear all;
clc;

fprintf('========================================================================\n');
fprintf('Identify Significant GPDC Connections via Surrogate Test\n');
fprintf('========================================================================\n\n');

% Set base path (modify as needed)
base_path = '/path/to/data/';

%% Load Data

fprintf('Loading real and surrogate GPDC data from Step 6...\n');

% Load from Step 6 output
load(fullfile(base_path, 'data_read_surr_gpdc2.mat'), 'data_surr', 'data');

% data: N_obs × (9 demographic + 972 GPDC values)
% data_surr: cell array (1000 surrogates) of same structure

fprintf('Real data: %d observations\n', size(data, 1));
fprintf('Surrogate datasets: %d\n\n', length(data_surr));

%% Define Connection Indices

% GPDC connectivity matrix organization:
% Columns 10-981: connectivity values
%   - 81 connections × 4 connection types × 3 frequency bands
%   - Connection types: II, AA, AI, IA
%   - Frequency bands: Delta (1-3 Hz), Theta (3-6 Hz), Alpha (6-9 Hz)

% Define indices for each connection type and frequency band
ii_delta = 10:90;      % Infant-Infant Delta
ii_theta = 91:171;     % Infant-Infant Theta
ii_alpha = 172:252;    % Infant-Infant Alpha
aa_delta = 253:333;    % Adult-Adult Delta
aa_theta = 334:414;    % Adult-Adult Theta
aa_alpha = 415:495;    % Adult-Adult Alpha
ai_delta = 496:576;    % Adult-Infant Delta
ai_theta = 577:657;    % Adult-Infant Theta
ai_alpha = 658:738;    % Adult-Infant Alpha
ia_delta = 739:819;    % Infant-Adult Delta
ia_theta = 820:900;    % Infant-Adult Theta
ia_alpha = 901:981;    % Infant-Adult Alpha

fprintf('GPDC index structure:\n');
fprintf('  II (Infant-Infant): Delta %d:%d, Theta %d:%d, Alpha %d:%d\n', ...
    ii_delta(1), ii_delta(end), ii_theta(1), ii_theta(end), ii_alpha(1), ii_alpha(end));
fprintf('  AA (Adult-Adult): Delta %d:%d, Theta %d:%d, Alpha %d:%d\n', ...
    aa_delta(1), aa_delta(end), aa_theta(1), aa_theta(end), aa_alpha(1), aa_alpha(end));
fprintf('  AI (Adult-Infant): Delta %d:%d, Theta %d:%d, Alpha %d:%d\n', ...
    ai_delta(1), ai_delta(end), ai_theta(1), ai_theta(end), ai_alpha(1), ai_alpha(end));
fprintf('  IA (Infant-Adult): Delta %d:%d, Theta %d:%d, Alpha %d:%d\n\n', ...
    ia_delta(1), ia_delta(end), ia_theta(1), ia_theta(end), ia_alpha(1), ia_alpha(end));

%% Process Each Connection Type and Band

% Define connection types to process
% For main analysis, we focus on II and AI alpha connections
connection_sets = {
    'II', ii_alpha;
    'AI', ai_alpha;
    'AA', aa_alpha;
};

fprintf('Processing connection types for alpha band:\n');
fprintf('  II (Infant-Infant)\n');
fprintf('  AI (Adult-Infant)\n');
fprintf('  AA (Adult-Adult)\n\n');

%% Loop Through Connection Types

for set_idx = 1:size(connection_sets, 1)

    conn_type = connection_sets{set_idx, 1};
    listi = connection_sets{set_idx, 2};

    fprintf('========================================================================\n');
    fprintf('Processing: %s Alpha Connections\n', conn_type);
    fprintf('========================================================================\n\n');

    fprintf('Testing %d connections (%s alpha band)\n', length(listi), conn_type);

    %% Apply Square Root Transform

    % Apply sqrt transform to stabilize variance (as in main analysis)
    data_real = sqrt(data(:, listi));

    %% Test Significance: Pooled Across Conditions

    fprintf('Computing mean connectivity across all observations...\n');

    % Real data mean for each connection
    mean_real = nanmean(data_real, 1);  % 1 × 81

    % Surrogate data means
    n_surr = length(data_surr);
    mean_surr = zeros(n_surr, length(listi));

    for surr_idx = 1:n_surr
        if mod(surr_idx, 200) == 0
            fprintf('  Processing surrogate %d/%d\n', surr_idx, n_surr);
        end

        data_surr_tmp = sqrt(data_surr{surr_idx}(:, listi));
        mean_surr(surr_idx, :) = nanmean(data_surr_tmp, 1);
    end

    fprintf('Surrogate mean calculation completed.\n\n');

    %% Compute P-values

    fprintf('Computing p-values via surrogate test...\n');

    p_values = zeros(length(listi), 1);

    for conn_idx = 1:length(listi)
        % One-tailed test: real > surrogate?
        % P-value = (# surrogates >= real + 1) / (# surrogates + 1)
        p_values(conn_idx) = (sum(mean_surr(:, conn_idx) >= mean_real(conn_idx)) + 1) / ...
                             (n_surr + 1);
    end

    fprintf('P-value calculation completed.\n\n');

    %% FDR Correction

    fprintf('Applying FDR correction (Benjamini-Hochberg)...\n');

    % FDR correction using MATLAB's mafdr function
    p_fdr = mafdr(p_values, 'BHFDR', true);

    % Identify significant connections at different thresholds
    alpha_fdr = 0.05;
    significant_idx = find(p_fdr < alpha_fdr);

    fprintf('Significant connections at FDR q < %.2f: %d / %d (%.1f%%)\n\n', ...
        alpha_fdr, length(significant_idx), length(listi), ...
        100*length(significant_idx)/length(listi));

    %% Save Results

    % Store indices of significant connections
    % These will be used in Step 11 for PLS regression
    s1 = [];  % Placeholder for Condition 1 (Full gaze)
    s2 = [];  % Placeholder for Condition 2 (Partial gaze)
    s3 = [];  % Placeholder for Condition 3 (No gaze)
    s4 = significant_idx;  % Significant pooled across conditions

    % Save stronglist
    save_filename = fullfile(base_path, sprintf('stronglistfdr5_gpdc_%s.mat', conn_type));
    save(save_filename, 'significant_idx', 's1', 's2', 's3', 's4', ...
         'p_values', 'p_fdr', 'mean_real', 'alpha_fdr');

    fprintf('Results saved to: %s\n', save_filename);
    fprintf('Variable s4 contains %d significant connection indices\n\n', length(s4));

    %% Display Significant Connections

    if ~isempty(significant_idx)
        fprintf('Significant %s connections (top 10):\n', conn_type);

        % Channel labels for interpretation
        ch_labels = {'F3','Fz','F4','C3','Cz','C4','P3','Pz','P4'};

        % Display up to first 10 significant connections
        n_display = min(10, length(significant_idx));
        for i = 1:n_display
            conn_idx = significant_idx(i);

            % Decode connection: sender and receiver
            % Connection index within 9×9 matrix
            sender = floor((conn_idx-1) / 9) + 1;
            receiver = mod(conn_idx-1, 9) + 1;

            fprintf('  %d. %s → %s: pFDR = %.4f, mean = %.4f\n', ...
                i, ch_labels{sender}, ch_labels{receiver}, ...
                p_fdr(conn_idx), mean_real(conn_idx));
        end

        if length(significant_idx) > 10
            fprintf('  ... and %d more connections\n', length(significant_idx) - 10);
        end
        fprintf('\n');
    else
        fprintf('No significant connections found.\n\n');
    end

    %% Summary for This Connection Type

    fprintf('Summary for %s alpha:\n', conn_type);
    fprintf('  Total connections: %d\n', length(listi));
    fprintf('  Significant (FDR q < %.2f): %d\n', alpha_fdr, length(significant_idx));
    fprintf('  Mean p-value: %.4f\n', mean(p_values));
    fprintf('  Min p-value: %.4f\n\n', min(p_values));

end

%% Overall Summary

fprintf('========================================================================\n');
fprintf('OVERALL SUMMARY\n');
fprintf('========================================================================\n\n');

fprintf('Feature Selection Method:\n');
fprintf('  - Based on surrogate test (real > chance baseline)\n');
fprintf('  - Independent of learning outcomes\n');
fprintf('  - Non-circular by design\n\n');

fprintf('Files generated:\n');
fprintf('  - stronglistfdr5_gpdc_II.mat\n');
fprintf('  - stronglistfdr5_gpdc_AI.mat\n');
fprintf('  - stronglistfdr5_gpdc_AA.mat\n\n');

fprintf('Each file contains:\n');
fprintf('  - s4: Indices of significant connections (main variable)\n');
fprintf('  - p_values: Uncorrected p-values\n');
fprintf('  - p_fdr: FDR-corrected p-values\n');
fprintf('  - mean_real: Mean GPDC for each connection\n\n');

fprintf('Next steps:\n');
fprintf('  - Step 11: Use stronglist files for PLS regression\n');
fprintf('  - These significant connections were selected WITHOUT using learning data\n');
fprintf('  - This ensures non-circular feature selection for subsequent prediction\n\n');

fprintf('========================================================================\n');
fprintf('Script complete.\n');
fprintf('========================================================================\n');
