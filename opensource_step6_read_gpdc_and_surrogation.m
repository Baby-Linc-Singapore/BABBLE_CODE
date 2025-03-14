%% EEG Connectivity Data Analysis
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.
%
% Purpose: Compare real connectivity data with surrogate data to assess statistical significance
%
% This script:
% 1. Loads behavioral data and connectivity matrices (GPDC)
% 2. Extracts connectivity measures across frequency bands
% 3. Loads surrogate connectivity data for comparison
% 4. Performs statistical analysis to identify significant connectivity patterns

%% Initialize environment
clc
clear all

% Set base path
base_path = '/path/to/data/';

% Select connectivity measure to analyze
connectivity_type = 'GPDC';  % Options: DC, DTF, PDC, GPDC, COH, PCOH

%% Load behavioral data 

% Load behavioral data (learning scores, attention, etc.)
[behavioral_data, ~] = xlsread(fullfile(base_path, 'table', 'behavioral_data.xlsx'));

% Remove entries with missing learning scores
behavioral_data(find(sum(isnan(behavioral_data(:,7)),2)>0),:) = [];

% Load window information
load(fullfile(base_path, 'code', 'final', 'surr', 'windowlist.mat'), 'non_empty_positionslist');

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

%% Process connectivity data

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
                file_path = fullfile(base_path, 'data_matfile', [connectivity_type, '3_nonorpdc_nonorpower'], 
                                   ['UK_', p_num, '_PDC.mat']);
            else
                % Location 2 (SG)
                file_path = fullfile(base_path, 'data_matfile', [connectivity_type, '3_nonorpdc_nonorpower'], 
                                   ['SG_', p_num, '_PDC.mat']);
            end
            
            % Load connectivity data
            try
                load(file_path);
                
                % Get current block and condition
                block = behavioral_data(i, 5);
                cond = behavioral_data(i, 6);
                
                % Check if sufficient windows exist for this participant/condition/block
                y = find(participant_list == behavioral_data(i, 1)*1000 + str2double(p_num));
                tmp = non_empty_positionslist{y};
                list = intersect(find(tmp(:,1) == block), find(tmp(:,2) == cond));
                sumwindow = sum(tmp(list, 4));
                
                if sumwindow > 0
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
                end
            catch
                fprintf('Error loading data for participant %d, block %d, condition %d\n', 
                       p_id, block, cond);
            end
        end
    end
end

fprintf('Processed connectivity data for %d valid data points\n', count);

%% Analyze connectivity patterns by type

% Define node matrix size
num_nodes = 9;  % 9 electrodes in grid montage

% Extract connectivity matrices by type and frequency band
segment1 = data(:, 10+(num_nodes^2)*0 : 9+(num_nodes^2)*1);  % II connectivity
segment2 = data(:, 10+(num_nodes^2)*1 : 9+(num_nodes^2)*2);  % AA connectivity
segment3 = data(:, 10+(num_nodes^2)*2 : 9+(num_nodes^2)*3);  % AI connectivity
segment4 = data(:, 10+(num_nodes^2)*3 : 9+(num_nodes^2)*4);  % IA connectivity

% Vectorize each segment
segment1 = segment1(:);
segment2 = segment2(:);
segment3 = segment3(:);
segment4 = segment4(:);

% Calculate summary statistics
result = [
    [nanmean(segment1), nanstd(segment1), length(segment1)];
    [nanmean(segment2), nanstd(segment2), length(segment2)];
    [nanmean(segment3), nanstd(segment3), length(segment3)];
    [nanmean(segment4), nanstd(segment4), length(segment4)]
];

% Display results table
fprintf('\nConnectivity summary by quadrant:\n');
fprintf('Quadrant\tMean\t\tStd\t\tN\n');
fprintf('II\t\t%.6f\t%.6f\t%d\n', result(1,1), result(1,2), result(1,3));
fprintf('AA\t\t%.6f\t%.6f\t%d\n', result(2,1), result(2,2), result(2,3));
fprintf('AI\t\t%.6f\t%.6f\t%d\n', result(3,1), result(3,2), result(3,3));
fprintf('IA\t\t%.6f\t%.6f\t%d\n', result(4,1), result(4,2), result(4,3));

% Compare Adult-to-Infant vs Infant-to-Infant connectivity
[h, p] = ttest2(segment3, segment1);
fprintf('\nComparison of AI vs II connectivity: t(%d) = %.2f, p = %.4f\n', 
       length(segment1) - 2, p.tstat, p);

%% Load surrogate connectivity data for comparison

% Define surrogate path prefix
surrogate_path_prefix = fullfile(base_path, 'data_matfile', ['surrPDCSET5/PDC']);

% Initialize storage for surrogate data
surrogate_data = cell(1000, 1);
count = 0;
surrogate_valid = zeros(1000, 1);

% Load behavioral data once
a = data;

% Process each surrogate iteration
for surr_idx = 1:1000
    fprintf('Processing surrogate %d of 1000\n', surr_idx);
    
    % Define path for current surrogate
    surrogate_path = [surrogate_path_prefix, num2str(surr_idx)];
    
    % Check if sufficient files exist
    files = dir(fullfile(surrogate_path, '*.mat'));
    if length(files) >= 42
        % Initialize temporary storage
        temp = zeros(size(data));
        count = count + 1;
        surrogate_valid(surr_idx) = 1;
        count1 = 0;
        
        % Process data for each participant
        for i = 1:size(a, 1)
            if ~isnan(a(i, 7))
                p_id = a(i, 2);
                
                % Skip excluded participants
                if ~ismember(p_id, [1112, 1116, 2112, 2118, 2119])
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
                        fprintf('Error loading surrogate data for participant %d in iteration %d\n', 
                               p_id, surr_idx);
                    end
                end
            end
        end
        
        % Store surrogate data for this iteration
        surrogate_data{count} = temp;
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

% Save processed data
save_path = fullfile(base_path, ['data_read_surr_', connectivity_type, '.mat']);
save(save_path, 'surrogate_data', 'data');

fprintf('\nProcessed %d valid surrogate iterations\n', sum(surrogate_valid));

%% Compare real vs surrogate connectivity

% Initialize arrays to store statistical results
mean_real = zeros(4, 3);  % 4 quadrants, 3 frequency bands
mean_surr = zeros(4, 3);  % 4 quadrants, 3 frequency bands
p_values = zeros(4, 3);   % 4 quadrants, 3 frequency bands

% Define quadrant names and frequency bands
quadrants = {'II', 'AA', 'AI', 'IA'};
bands = {'Delta', 'Theta', 'Alpha'};

% Extract means from real data
for q = 1:4  % Quadrants
    for f = 1:3  % Frequency bands
        % Calculate column indices for this quadrant/frequency
        start_idx = 10 + (q-1)*(num_nodes^2)*3 + (f-1)*num_nodes^2;
        end_idx = start_idx + num_nodes^2 - 1;
        
        % Extract connectivity values
        connectivity_values = data(:, start_idx:end_idx);
        connectivity_values = connectivity_values(:);
        
        % Store mean
        mean_real(q, f) = nanmean(connectivity_values);
    end
end

% Calculate means and p-values for each surrogate iteration
for s = 1:length(surrogate_data)
    surr = surrogate_data{s};
    
    % Process each quadrant and frequency band
    for q = 1:4  % Quadrants
        for f = 1:3  % Frequency bands
            % Calculate column indices for this quadrant/frequency
            start_idx = 10 + (q-1)*(num_nodes^2)*3 + (f-1)*num_nodes^2;
            end_idx = start_idx + num_nodes^2 - 1;
            
            % Extract connectivity values
            connectivity_values = surr(:, start_idx:end_idx);
            connectivity_values = connectivity_values(:);
            
            % Compare with real data
            [~, p] = ttest2(data(:, start_idx:end_idx), connectivity_values);
            
            % Accumulate surrogate means
            mean_surr(q, f) = mean_surr(q, f) + nanmean(connectivity_values);
            
            % Count significant results
            if p < 0.05
                p_values(q, f) = p_values(q, f) + 1;
            end
        end
    end
end

% Average surrogate means
mean_surr = mean_surr / length(surrogate_data);

% Calculate proportion of significant results
p_values = p_values / length(surrogate_data);

% Display results
fprintf('\nComparison of real vs surrogate connectivity by quadrant and frequency band:\n');
fprintf('Quadrant\tFrequency\tReal Mean\tSurr Mean\tP(Real≠Surr)\n');
for q = 1:4
    for f = 1:3
        fprintf('%s\t\t%s\t\t%.6f\t%.6f\t%.2f%%\n', ...
               quadrants{q}, bands{f}, mean_real(q, f), mean_surr(q, f), p_values(q, f)*100);
    end
end

fprintf('\nAnalysis complete.\n');