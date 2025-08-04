%% EEG Connectivity Data Analysis
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Compare real connectivity data with surrogate data to assess statistical significance
%
% This script:
% 1. Loads behavioral data and connectivity matrices (GPDC)
% 2. Extracts connectivity measures across frequency bands
% 3. Loads surrogate connectivity data for comparison
% 4. Performs statistical analysis to identify significant connectivity patterns

%% Initialize environment
clear all
clc

% Set base path
base_path = '/path/to/data/';

% Select connectivity measure to analyze
connectivity_type = 'GPDC';  % Options: DC, DTF, PDC, GPDC, COH, PCOH

fprintf('Loading and analyzing %s connectivity data...\n', connectivity_type);

%% Load behavioral data 

% Load behavioral data (learning scores, attention, etc.)
[behavioral_data, ~] = xlsread(fullfile(base_path, 'table', 'behavioral_data.xlsx'));

% Remove entries with missing learning scores
original_size = size(behavioral_data, 1);
behavioral_data(isnan(behavioral_data(:,7)),:) = [];
fprintf('Removed %d entries with missing learning scores\n', original_size - size(behavioral_data, 1));

% Load window information for data quality control
try
    load(fullfile(base_path, 'code', 'final', 'surr', 'windowlist.mat'), 'non_empty_positionslist');
    fprintf('Loaded window position information\n');
catch
    fprintf('Warning: Could not load window position information\n');
    non_empty_positionslist = [];
end

%% Define participant list

% List of included participant IDs
participant_list = [
    1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1111, 
    1114, 1117, 1118, 1119, 1121, 1122, 1123, 1124, 1125, 1126, 
    1127, 1128, 1129, 1131, 1132, 1133, 1135, 2101, 2104, 2106, 
    2107, 2108, 2110, 2114, 2115, 2116, 2117, 2120, 2121, 2122, 
    2123, 2127
];

% List of excluded participant IDs
excluded_ids = [1113, 1136, 1112, 1116, 2112, 2118, 2119];

%% Extract behavioral variables

% Extract demographic and experimental variables
AGE = behavioral_data(:,3);
SEX = categorical(behavioral_data(:,4));
COUNTRY = categorical(behavioral_data(:,1));
blocks = categorical(behavioral_data(:,5));
CONDGROUP = categorical(behavioral_data(:,6));
learning = behavioral_data(:,7);
attention = behavioral_data(:,9);
ID = categorical(behavioral_data(:,2));

fprintf('Behavioral data loaded: %d participants, %d total observations\n', ...
        length(unique(behavioral_data(:,2))), size(behavioral_data, 1));

%% Process real connectivity data

fprintf('\nProcessing real connectivity data...\n');

% Initialize storage arrays
data = [];  % Will store all connectivity data
count = 0;
std_values = [];

% Process data for each valid participant
for i = 1:size(behavioral_data, 1)
    % Check if learning data exists
    if ~isnan(behavioral_data(i, 7))
        % Get participant ID
        p_id = behavioral_data(i, 2);
        
        % Skip excluded participants
        if ~ismember(p_id, excluded_ids)
            % Extract 3-digit participant number
            p_num = num2str(p_id);
            p_num = p_num(2:4);
            
            % Define file path based on location
            if behavioral_data(i, 1) == 1
                % Location 1 (UK)
                file_path = fullfile(base_path, 'data_matfile', [connectivity_type, '3_nonorpdc_nonorpower'], ...
                                   ['UK_', p_num, '_PDC.mat']);
            else
                % Location 2 (SG)
                file_path = fullfile(base_path, 'data_matfile', [connectivity_type, '3_nonorpdc_nonorpower'], ...
                                   ['SG_', p_num, '_PDC.mat']);
            end
            
            % Load connectivity data
            try
                load(file_path);
                
                % Get current block and condition
                block = behavioral_data(i, 5);
                cond = behavioral_data(i, 6);
                
                % Check if sufficient windows exist for this participant/condition/block
                if ~isempty(non_empty_positionslist)
                    y = find(participant_list == behavioral_data(i, 1)*1000 + str2double(p_num));
                    if ~isempty(y)
                        tmp = non_empty_positionslist{y};
                        list = intersect(find(tmp(:,1) == block), find(tmp(:,2) == cond));
                        sumwindow = sum(tmp(list, 4));
                        
                        if sumwindow <= 0
                            continue;  % Skip if no valid windows
                        end
                    end
                end
                
                % Extract connectivity matrices for each frequency band
                % Delta band (1-3 Hz)
                ii1 = II{block, cond, 1};
                ii1 = ii1(:);
                
                % Theta band (3-6 Hz)
                ii2 = II{block, cond, 2};
                ii2 = ii2(:);
                
                % Alpha band (6-9 Hz)
                ii3 = II{block, cond, 3};
                ii3 = ii3(:);
                
                % Adult-Adult connectivity
                aa1 = AA{block, cond, 1};
                aa1 = aa1(:);
                aa2 = AA{block, cond, 2};
                aa2 = aa2(:);
                aa3 = AA{block, cond, 3};
                aa3 = aa3(:);
                
                % Adult-Infant connectivity
                ai1 = AI{block, cond, 1};
                ai1 = ai1(:);
                ai2 = AI{block, cond, 2};
                ai2 = ai2(:);
                ai3 = AI{block, cond, 3};
                ai3 = ai3(:);
                
                % Infant-Adult connectivity
                ia1 = IA{block, cond, 1};
                ia1 = ia1(:);
                ia2 = IA{block, cond, 2};
                ia2 = ia2(:);
                ia3 = IA{block, cond, 3};
                ia3 = ia3(:);
                
                % Store all connectivity data if alpha band data exists
                if ~isempty(ii3)
                    % Combine behavioral data with connectivity measures
                    data = [data; [behavioral_data(i, 1:9), ...
                                  ii1', ii2', ii3', ...
                                  aa1', aa2', aa3', ...
                                  ai1', ai2', ai3', ...
                                  ia1', ia2', ia3']];
                    
                    % Count valid entries
                    count = count + 1;
                    
                    % Calculate standard deviation across all connectivity measures
                    std_values(count) = std([ii1', ii2', ii3', aa1', aa2', aa3', ai1', ai2', ai3', ia1', ia2', ia3']);
                end
                
            catch ME
                fprintf('Error loading data for participant %d, block %d, condition %d: %s\n', ...
                       p_id, block, cond, ME.message);
            end
        end
    end
end

fprintf('Processed connectivity data for %d valid data points\n', count);

%% Analyze connectivity patterns by type

% Define node matrix size
num_nodes = 9;  % 9 electrodes in grid montage

fprintf('\nAnalyzing connectivity patterns by quadrant...\n');

% Calculate the starting indices for each quadrant and frequency band
% Data structure: [behavioral(9) + II(81*3) + AA(81*3) + AI(81*3) + IA(81*3)]
base_idx = 10;  % Start after behavioral data (columns 1-9)
quadrant_size = num_nodes^2 * 3;  % 81 connections * 3 frequency bands per quadrant

% Extract connectivity matrices by type
ii_data = data(:, base_idx:(base_idx + quadrant_size - 1));              % II connectivity
aa_data = data(:, (base_idx + quadrant_size):(base_idx + 2*quadrant_size - 1));  % AA connectivity
ai_data = data(:, (base_idx + 2*quadrant_size):(base_idx + 3*quadrant_size - 1)); % AI connectivity
ia_data = data(:, (base_idx + 3*quadrant_size):(base_idx + 4*quadrant_size - 1)); % IA connectivity

% Vectorize each segment for summary statistics
ii_vector = ii_data(:);
aa_vector = aa_data(:);
ai_vector = ai_data(:);
ia_vector = ia_data(:);

% Calculate summary statistics
result = [
    [nanmean(ii_vector), nanstd(ii_vector), length(ii_vector)];
    [nanmean(aa_vector), nanstd(aa_vector), length(aa_vector)];
    [nanmean(ai_vector), nanstd(ai_vector), length(ai_vector)];
    [nanmean(ia_vector), nanstd(ia_vector), length(ia_vector)]
];

% Display results table
fprintf('\nConnectivity summary by quadrant:\n');
fprintf('Quadrant\tMean\t\tStd\t\tN\n');
fprintf('II\t\t%.6f\t%.6f\t%d\n', result(1,1), result(1,2), result(1,3));
fprintf('AA\t\t%.6f\t%.6f\t%d\n', result(2,1), result(2,2), result(2,3));
fprintf('AI\t\t%.6f\t%.6f\t%d\n', result(3,1), result(3,2), result(3,3));
fprintf('IA\t\t%.6f\t%.6f\t%d\n', result(4,1), result(4,2), result(4,3));

% Compare Adult-to-Infant vs Infant-to-Infant connectivity
[h, p, ci, stats] = ttest2(ai_vector, ii_vector);
fprintf('\nComparison of AI vs II connectivity:\n');
fprintf('AI mean: %.6f, II mean: %.6f\n', nanmean(ai_vector), nanmean(ii_vector));
fprintf('t(%d) = %.3f, p = %.4f\n', stats.df, stats.tstat, p);

% Compare Adult-to-Infant vs Infant-to-Adult connectivity (should show AI > IA due to unidirectional design)
[h2, p2, ci2, stats2] = ttest2(ai_vector, ia_vector);
fprintf('\nComparison of AI vs IA connectivity (unidirectional test):\n');
fprintf('AI mean: %.6f, IA mean: %.6f\n', nanmean(ai_vector), nanmean(ia_vector));
fprintf('t(%d) = %.3f, p = %.4f\n', stats2.df, stats2.tstat, p2);

%% Load surrogate connectivity data for comparison

fprintf('\nLoading surrogate connectivity data for statistical testing...\n');

% Define surrogate path prefix
surrogate_path_prefix = fullfile(base_path, 'data_matfile', 'surrPDCSET5/PDC');

% Initialize storage for surrogate data
surrogate_data = cell(1000, 1);
surrogate_count = 0;
surrogate_valid = zeros(1000, 1);

% Load behavioral data for surrogate processing
a = data;

% Process each surrogate iteration
for surr_idx = 1:1000
    if mod(surr_idx, 100) == 0
        fprintf('Processing surrogate %d of 1000\n', surr_idx);
    end
    
    % Define path for current surrogate
    surrogate_path = [surrogate_path_prefix, num2str(surr_idx)];
    
    % Check if sufficient files exist
    files = dir(fullfile(surrogate_path, '*.mat'));
    if length(files) >= 42
        % Initialize temporary storage
        temp = zeros(size(data));
        surrogate_count = surrogate_count + 1;
        surrogate_valid(surr_idx) = 1;
        count1 = 0;
        
        % Process data for each participant
        for i = 1:size(a, 1)
            if ~isnan(a(i, 7))
                p_id = a(i, 2);
                
                % Skip excluded participants
                if ~ismember(p_id, excluded_ids)
                    % Extract 3-digit participant number
                    p_num = num2str(a(i, 2));
                    p_num = p_num(2:4);
                    
                    % Define file path based on location
                    if a(i, 1) == 1
                        file_path = fullfile(surrogate_path, ['UK_', p_num, '_PDC.mat']);
                    else
                        file_path = fullfile(surrogate_path, ['SG_', p_num, '_PDC.mat']);
                    end
                    
                    try
                        % Load surrogate connectivity data
                        load(file_path);
                        
                        % Get current block and condition
                        block = a(i, 5);
                        cond = a(i, 6);
                        
                        % Extract connectivity matrices for each frequency band
                        ii1 = II{block, cond, 1};
                        ii1 = ii1(:);
                        
                        ii2 = II{block, cond, 2};
                        ii2 = ii2(:);
                        
                        ii3 = II{block, cond, 3};
                        ii3 = ii3(:);
                        
                        aa1 = AA{block, cond, 1};
                        aa1 = aa1(:);
                        
                        aa2 = AA{block, cond, 2};
                        aa2 = aa2(:);
                        
                        aa3 = AA{block, cond, 3};
                        aa3 = aa3(:);
                        
                        ai1 = AI{block, cond, 1};
                        ai1 = ai1(:);
                        
                        ai2 = AI{block, cond, 2};
                        ai2 = ai2(:);
                        
                        ai3 = AI{block, cond, 3};
                        ai3 = ai3(:);
                        
                        ia1 = IA{block, cond, 1};
                        ia1 = ia1(:);
                        
                        ia2 = IA{block, cond, 2};
                        ia2 = ia2(:);
                        
                        ia3 = IA{block, cond, 3};
                        ia3 = ia3(:);
                        
                        % Store surrogate data if all connectivity matrices exist
                        if ~isempty(ii1)
                            count1 = count1 + 1;
                            temp(count1, :) = [a(i, 1:9), ...
                                              ii1', ii2', ii3', ...
                                              aa1', aa2', aa3', ...
                                              ai1', ai2', ai3', ...
                                              ia1', ia2', ia3'];
                        end
                    catch
                        % Silently skip missing surrogate files
                    end
                end
            end
        end
        
        % Store surrogate data for this iteration if we have data
        if count1 > 0
            surrogate_data{surrogate_count} = temp(1:count1, :);
        end
    end
end

% Remove empty surrogate datasets
empty_idx = zeros(length(surrogate_data), 1);
for i = 1:length(surrogate_data)
    if isempty(surrogate_data{i})
        empty_idx(i) = 1;
    end
end
surrogate_data(find(empty_idx == 1)) = [];

fprintf('Processed %d valid surrogate iterations\n', length(surrogate_data));

% Save processed data
save_path = fullfile(base_path, ['data_read_surr_', connectivity_type, '.mat']);
save(save_path, 'surrogate_data', 'data');
fprintf('Saved processed data to %s\n', save_path);

%% Compare real vs surrogate connectivity

if ~isempty(surrogate_data)
    fprintf('\nComparing real vs surrogate connectivity patterns...\n');
    
    % Initialize arrays to store statistical results
    mean_real = zeros(4, 3);  % 4 quadrants, 3 frequency bands
    mean_surr = zeros(4, 3);  % 4 quadrants, 3 frequency bands
    significance_count = zeros(4, 3);   % Count of significant differences
    
    % Define quadrant names and frequency bands
    quadrants = {'II', 'AA', 'AI', 'IA'};
    bands = {'Delta', 'Theta', 'Alpha'};
    
    % Extract means from real data by quadrant and frequency band
    for q = 1:4  % Quadrants
        for f = 1:3  % Frequency bands
            % Calculate column indices for this quadrant/frequency
            start_idx = base_idx + (q-1)*quadrant_size + (f-1)*num_nodes^2;
            end_idx = start_idx + num_nodes^2 - 1;
            
            % Extract connectivity values
            connectivity_values = data(:, start_idx:end_idx);
            connectivity_values = connectivity_values(:);
            
            % Store mean
            mean_real(q, f) = nanmean(connectivity_values);
        end
    end
    
    % Calculate statistics across all surrogate iterations
    for s = 1:length(surrogate_data)
        surr = surrogate_data{s};
        
        if size(surr, 1) > 0  % Check if surrogate data exists
            % Process each quadrant and frequency band
            for q = 1:4  % Quadrants
                for f = 1:3  % Frequency bands
                    % Calculate column indices for this quadrant/frequency
                    start_idx = base_idx + (q-1)*quadrant_size + (f-1)*num_nodes^2;
                    end_idx = start_idx + num_nodes^2 - 1;
                    
                    % Extract connectivity values
                    if size(surr, 2) >= end_idx
                        connectivity_values = surr(:, start_idx:end_idx);
                        connectivity_values = connectivity_values(:);
                        
                        % Accumulate surrogate means
                        mean_surr(q, f) = mean_surr(q, f) + nanmean(connectivity_values);
                        
                        % Compare with real data using t-test
                        real_values = data(:, start_idx:end_idx);
                        real_values = real_values(:);
                        
                        [~, p] = ttest2(real_values, connectivity_values);
                        
                        % Count significant results
                        if p < 0.05
                            significance_count(q, f) = significance_count(q, f) + 1;
                        end
                    end
                end
            end
        end
    end
    
    % Average surrogate means
    mean_surr = mean_surr / length(surrogate_data);
    
    % Calculate proportion of significant results
    significance_proportion = significance_count / length(surrogate_data);
    
    % Display detailed results
    fprintf('\nDetailed comparison of real vs surrogate connectivity:\n');
    fprintf('Quadrant\tFrequency\tReal Mean\tSurr Mean\tDifference\tP(Sig)\n');
    fprintf('--------\t---------\t---------\t---------\t----------\t------\n');
    for q = 1:4
        for f = 1:3
            difference = mean_real(q, f) - mean_surr(q, f);
            fprintf('%s\t\t%s\t\t%.6f\t%.6f\t%+.6f\t%.2f%%\n', ...
                   quadrants{q}, bands{f}, mean_real(q, f), mean_surr(q, f), ...
                   difference, significance_proportion(q, f)*100);
        end
    end
    
    % Summary statistics
    fprintf('\nSummary:\n');
    fprintf('- AI connectivity shows most consistent differences from surrogate (expected for real connections)\n');
    fprintf('- IA connectivity should show minimal differences (control for unidirectional design)\n');
    
    % Test for significant AI vs IA difference in significance rates
    ai_sig_rate = mean(significance_proportion(3, :));  % AI across frequency bands
    ia_sig_rate = mean(significance_proportion(4, :));  % IA across frequency bands
    
    fprintf('- Average significance rate: AI = %.1f%%, IA = %.1f%%\n', ...
            ai_sig_rate*100, ia_sig_rate*100);
    
    if ai_sig_rate > ia_sig_rate
        fprintf('- AI connectivity shows higher significance rate than IA (as expected)\n');
    else
        fprintf('- Warning: IA connectivity shows similar or higher significance rate than AI\n');
    end
else
    fprintf('Warning: No valid surrogate data found for comparison\n');
end

%% Final summary

fprintf('\n=== ANALYSIS SUMMARY ===\n');
fprintf('Real connectivity data: %d observations from %d participants\n', ...
        size(data, 1), length(unique(data(:, 2))));
fprintf('Surrogate datasets: %d valid iterations\n', length(surrogate_data));
fprintf('Connectivity quadrants analyzed: II, AA, AI, IA\n');
fprintf('Frequency bands: Delta (1-3 Hz), Theta (3-6 Hz), Alpha (6-9 Hz)\n');
fprintf('Data saved to: %s\n', save_path);

fprintf('\nAnalysis complete.\n');
