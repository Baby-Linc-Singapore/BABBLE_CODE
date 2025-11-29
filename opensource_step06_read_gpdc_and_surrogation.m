%% Aggregate GPDC Data from Individual Participants
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to
% NTU's open access policy.
%
% Purpose: Read individual participant GPDC files and compile into analysis matrix
%
% This script:
% 1. Reads behavioral data
% 2. Loads GPDC data from each participant's file (Step 4 output)
% 3. Combines into single matrix for statistical analysis
% 4. Saves aggregated data for subsequent steps
%
% Based on: scripts_R1/fs4_readdata.m

clear all;
clc;

fprintf('========================================================================\n');
fprintf('Aggregate GPDC Data from Individual Participants\n');
fprintf('========================================================================\n\n');

% Set base path (modify as needed)
base_path = '/path/to/data/';

%% Load Behavioral Data

fprintf('Loading behavioral data...\n');

behavioral_file = fullfile(base_path, 'behavioral_data.xlsx');
[behavioral_data, ~] = xlsread(behavioral_file);

% Remove observations with missing learning scores
n_original = size(behavioral_data, 1);
behavioral_data(isnan(behavioral_data(:,7)), :) = [];
n_valid = size(behavioral_data, 1);

fprintf('  Original observations: %d\n', n_original);
fprintf('  Valid observations (with learning data): %d\n\n', n_valid);

%% Define Participant Lists (Consistent Across All Scripts)

% EEG-valid UK participants (n=27)
uk_list = {'101', '102', '103', '104', '105', '106', '107', '108', '109', ...
           '111', '114', '117', '118', '119', '121', '122', '123', '124', ...
           '125', '126', '127', '128', '129', '131', '132', '133', '135'};

% EEG-valid SG participants (n=15)
sg_list = {'101', '104', '106', '107', '108', '110', '114', '115', ...
           '116', '117', '120', '121', '122', '123', '127'};

% Excluded participant IDs (n=7)
% Reasons: Excessive EEG artifacts or insufficient valid data windows
% (See Supplementary Materials for detailed exclusion criteria)
excluded_ids = [1113, 1136, 1112, 1116, 2112, 2118, 2119];

fprintf('Participant Inclusion Criteria:\n');
fprintf('  Total enrolled: 49 infants\n');
fprintf('  Behavioral data valid: 47 infants\n');
fprintf('  EEG data valid: 42 infants\n\n');

fprintf('Exclusion reasons:\n');
fprintf('  Behavioral exclusions (n=2): Insufficient attention/missing data\n');
fprintf('  EEG exclusions (n=7): Excessive artifacts (>50%% trials rejected)\n\n');

fprintf('Final sample for EEG analysis:\n');
fprintf('  UK site: %d participants\n', length(uk_list));
fprintf('  SG site: %d participants\n', length(sg_list));
fprintf('  Total: %d participants\n\n', length(uk_list) + length(sg_list));

%% Initialize Data Matrix

% Column structure (total: 981 columns):
%   1: Country (1=UK, 2=SG)
%   2: Subject ID
%   3: Age (days)
%   4: Sex (1=Male, 2=Female)
%   5: Block (1-3)
%   6: Condition (1=Full, 2=Partial, 3=No gaze)
%   7: Learning score (nonword - word looking time, seconds)
%   8: Learning score (duplicate for compatibility)
%   9: Attention proportion
%   10-981: GPDC connectivity values (972 values)
%     - II delta (81 connections): columns 10-90
%     - II theta (81 connections): columns 91-171
%     - II alpha (81 connections): columns 172-252
%     - AA delta (81 connections): columns 253-333
%     - AA theta (81 connections): columns 334-414
%     - AA alpha (81 connections): columns 415-495
%     - AI delta (81 connections): columns 496-576
%     - AI theta (81 connections): columns 577-657
%     - AI alpha (81 connections): columns 658-738
%     - IA delta (81 connections): columns 739-819
%     - IA theta (81 connections): columns 820-900
%     - IA alpha (81 connections): columns 901-981

n_connectivity_values = 81 * 4 * 3;  % 81 connections × 4 types × 3 bands
data = [];

fprintf('Data matrix structure:\n');
fprintf('  Columns 1-9: Demographics and behavioral measures\n');
fprintf('  Columns 10-981: GPDC connectivity (972 values)\n');
fprintf('    - 81 connections per type/band (9×9 channel pairs)\n');
fprintf('    - 4 types: II, AA, AI, IA\n');
fprintf('    - 3 bands: Delta, Theta, Alpha\n\n');

%% Read GPDC Data from Individual Files

fprintf('Reading GPDC data from individual participant files...\n');
fprintf('(This may take several minutes...)\n\n');

count = 0;
n_errors = 0;

for obs_idx = 1:size(behavioral_data, 1)

    % Skip if no learning data
    if isnan(behavioral_data(obs_idx, 7))
        continue;
    end

    % Extract participant info
    participant_id = behavioral_data(obs_idx, 2);

    % Skip excluded participants
    if ismember(participant_id, excluded_ids)
        continue;
    end

    % Get 3-digit participant number
    pid_str = num2str(participant_id);
    pid_num = pid_str(2:4);

    % Determine site and construct filename
    if behavioral_data(obs_idx, 1) == 1
        % UK site
        gpdc_file = fullfile(base_path, 'data_matfile', 'GPDC3_nonorpdc_nonorpower', ...
                            ['UK_', pid_num, '_PDC.mat']);
    else
        % SG site
        gpdc_file = fullfile(base_path, 'data_matfile', 'GPDC3_nonorpdc_nonorpower', ...
                            ['SG_', pid_num, '_PDC.mat']);
    end

    % Load GPDC data
    try
        % Load connectivity matrices
        load(gpdc_file, 'II', 'AA', 'AI', 'IA');

        % Extract block and condition
        block = behavioral_data(obs_idx, 5);
        cond = behavioral_data(obs_idx, 6);

        % Extract connectivity matrices for each band and reshape to vectors
        % Delta band (1-3 Hz)
        ii_delta = II{block, cond, 1}(:);  % 9×9 matrix → 81×1 vector
        aa_delta = AA{block, cond, 1}(:);
        ai_delta = AI{block, cond, 1}(:);
        ia_delta = IA{block, cond, 1}(:);

        % Theta band (3-6 Hz)
        ii_theta = II{block, cond, 2}(:);
        aa_theta = AA{block, cond, 2}(:);
        ai_theta = AI{block, cond, 2}(:);
        ia_theta = IA{block, cond, 2}(:);

        % Alpha band (6-9 Hz)
        ii_alpha = II{block, cond, 3}(:);
        aa_alpha = AA{block, cond, 3}(:);
        ai_alpha = AI{block, cond, 3}(:);
        ia_alpha = IA{block, cond, 3}(:);

        % Combine all connectivity values in proper order
        connectivity_row = [ii_delta', ii_theta', ii_alpha', ...
                           aa_delta', aa_theta', aa_alpha', ...
                           ai_delta', ai_theta', ai_alpha', ...
                           ia_delta', ia_theta', ia_alpha'];

        % Append to data matrix
        data = [data; behavioral_data(obs_idx, 1:9), connectivity_row];

        count = count + 1;

        % Progress indicator
        if mod(count, 20) == 0
            fprintf('  Processed %d observations...\n', count);
        end

    catch ME
        n_errors = n_errors + 1;
        if n_errors <= 5  % Only show first 5 errors
            fprintf('  Warning: Could not load data for participant %d, block %d, cond %d\n', ...
                    participant_id, block, cond);
            fprintf('    Error: %s\n', ME.message);
        end
    end
end

fprintf('\nData aggregation completed.\n');
fprintf('  Total observations processed: %d\n', count);
fprintf('  Errors encountered: %d\n', n_errors);
fprintf('  Data matrix dimensions: %d rows × %d columns\n\n', size(data, 1), size(data, 2));

%% Verify Data Integrity

fprintf('Verifying data integrity...\n');

% Check for expected number of participants
n_participants = length(unique(data(:,2)));
fprintf('  Unique participants: %d (expected: 42)\n', n_participants);

% Check observations per condition
for cond = 1:3
    n_obs_cond = sum(data(:,6) == cond);
    fprintf('  Condition %d observations: %d\n', cond, n_obs_cond);
end

% Check for NaN values in connectivity
n_nan_connectivity = sum(sum(isnan(data(:,10:end))));
total_connectivity_values = numel(data(:,10:end));
nan_percentage = 100 * n_nan_connectivity / total_connectivity_values;
fprintf('  NaN values in connectivity: %d / %d (%.2f%%)\n', ...
    n_nan_connectivity, total_connectivity_values, nan_percentage);

% Check learning score range
fprintf('  Learning score range: %.2f to %.2f seconds\n', ...
    min(data(:,7)), max(data(:,7)));
fprintf('  Learning score mean ± SD: %.2f ± %.2f seconds\n\n', ...
    mean(data(:,7)), std(data(:,7)));

%% Save Aggregated Data

fprintf('Saving aggregated data...\n');

output_file = fullfile(base_path, 'data_read_surr_gpdc2.mat');

% Note: This script only saves real data
% Surrogate data (data_surr) would be generated by a separate surrogate analysis script
% For now, we create a placeholder to maintain compatibility
data_surr = {};  % Placeholder - actual surrogates generated in separate script

save(output_file, 'data', 'data_surr', '-v7.3');

fprintf('Aggregated data saved to: %s\n\n', output_file);

%% Display Usage Information

fprintf('This file will be used in:\n');
fprintf('  - Step 5: Identify significant connections (surrogate test)\n');
fprintf('  - Step 11: PLS prediction analysis\n');
fprintf('  - Step 12: Mediation analysis\n');
fprintf('  - Step 18: Single-connection validation\n\n');

fprintf('Data structure reference:\n');
fprintf('  Load with: load(''data_read_surr_gpdc2.mat'', ''data'')\n');
fprintf('  Access country: data(:, 1)\n');
fprintf('  Access subject ID: data(:, 2)\n');
fprintf('  Access age: data(:, 3)\n');
fprintf('  Access sex: data(:, 4)\n');
fprintf('  Access block: data(:, 5)\n');
fprintf('  Access condition: data(:, 6)\n');
fprintf('  Access learning: data(:, 7)\n');
fprintf('  Access attention: data(:, 9)\n');
fprintf('  Access II alpha GPDC: data(:, 172:252)\n');
fprintf('  Access AI alpha GPDC: data(:, 658:738)\n');
fprintf('  Access AA alpha GPDC: data(:, 415:495)\n\n');

%% Summary

fprintf('========================================================================\n');
fprintf('DATA AGGREGATION SUMMARY\n');
fprintf('========================================================================\n\n');

fprintf('Input: Individual GPDC files (Step 4 output)\n');
fprintf('  Format: UK_###_PDC.mat, SG_###_PDC.mat\n');
fprintf('  Location: data_matfile/GPDC3_nonorpdc_nonorpower/\n\n');

fprintf('Output: Aggregated analysis matrix\n');
fprintf('  File: %s\n', output_file);
fprintf('  Dimensions: %d observations × %d variables\n', size(data, 1), size(data, 2));
fprintf('  Participants: %d (UK: %d, SG: %d)\n', n_participants, ...
    sum(data(:,1)==1)/sum(data(:,2)==data(1,2)), ...
    sum(data(:,1)==2)/sum(data(:,2)==data(1,2)));
fprintf('  Block-level observations: %d\n\n', size(data, 1));

fprintf('Next steps:\n');
fprintf('  1. Generate surrogate data (optional, for Step 5)\n');
fprintf('  2. Run Step 5 to identify significant connections\n');
fprintf('  3. Use aggregated data in subsequent analyses\n\n');

fprintf('========================================================================\n');
fprintf('Script complete.\n');
fprintf('========================================================================\n');
