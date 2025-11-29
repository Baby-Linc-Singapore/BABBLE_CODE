%% Learning and Attention Analysis Script
% NOTE: This code demonstrates the analytical methodology.
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to
% NTU's open access policy.
%
% Purpose: Calculate learning scores and attention metrics from raw behavioral data
%
% This script:
% 1. Processes looking time data from two datasets (locations)
% 2. Calculates learning scores as the difference between looking times for word1 and word2
% 3. Computes attention proportions from time-coded video data
% 4. Applies outlier exclusion criteria (default: 2.5 SD from mean)
% 5. Combines data into a structured format and exports to Excel

%% Initialize environment
clear all
clc

% Set base path (modify as needed)
base_path = '/path/to/data/';

% Outlier exclusion criterion
SD_cutoff = 2.5;  % Number of standard deviations for outlier detection

%% Process data from Location 1 (learning scores)

fprintf('Processing Location 1 learning data...\n');

% Load raw looking time data
try
    data_location1 = textread(fullfile(base_path, 'looktime/Location1_AllData.txt'));
catch
    data_location1 = textread(fullfile(base_path, 'looktime/Location1_AllData.txt'));
end

% Get unique participant IDs
participants_location1 = unique(data_location1(:,1));
participants_location1_ID = unique(data_location1(:,1)) + 1000;

% Remove excluded participants (IDs 12 and 31 based on fs1 code)
exclude_idx = [12, 31];
participants_location1(exclude_idx) = [];
participants_location1_ID(exclude_idx) = [];

% Initialize counters for data cleaning statistics
total_trials = 0;
clean_trials = 0;

% Initialize arrays for demographic information
age_location1 = zeros(length(participants_location1_ID), 1);
sex_location1 = zeros(length(participants_location1_ID), 1);

% Initialize cell array for mean looking times
MeanLook_location1 = cell(1, 3);  % 3 conditions
for cond = 1:3
    MeanLook_location1{cond} = NaN(length(participants_location1_ID), 2, 3);  % participants × words × blocks
end

% Process each participant
for p = 1:length(participants_location1_ID)
    % Find valid trials (non-zero looking times)
    valid_trials = find(data_location1(:,1) == participants_location1(p) & data_location1(:,9) > 0);
    
    if ~isempty(valid_trials)
        % Extract demographic information
        age_location1(p) = data_location1(valid_trials(1), 2);
        sex_location1(p) = data_location1(valid_trials(1), 3);
        
        % Extract trial information
        % Columns: 6=condition, 8=word, 9=looking time, 10=error flag, 5=block
        trial_data = [data_location1(valid_trials, 6), data_location1(valid_trials, 8), ...
                     data_location1(valid_trials, 9), data_location1(valid_trials, 10), ... 
                     data_location1(valid_trials, 5)];
        
        % Calculate individual outlier threshold
        outlier_threshold = mean(trial_data(:,3)) + SD_cutoff * std(trial_data(:,3));
        
        % Process each block
        for block = 1:3
            % Find trials within outlier threshold for this block
            valid_block_trials = find(trial_data(:,3) < outlier_threshold & trial_data(:,5) == block);
            
            % Check if there are enough valid trials and both words are represented
            if length(valid_block_trials) > 1 && length(unique(trial_data(valid_block_trials, 2))) == 2
                % Process each condition
                for cond = 1:3
                    % Process each word
                    for word = 1:2
                        % Find valid trials for this condition, word, and block
                        valid_trials_subset = find(trial_data(:,1) == cond & trial_data(:,2) == word & ...
                                                  trial_data(:,3) < outlier_threshold & trial_data(:,5) == block);
                        
                        % Count all trials for this condition, word, and block (including outliers)
                        all_trials_subset = find(trial_data(:,1) == cond & trial_data(:,2) == word & trial_data(:,5) == block);
                        
                        % Calculate mean looking time if valid trials exist
                        if ~isempty(valid_trials_subset)
                            MeanLook_location1{cond}(p, word, block) = nanmean(trial_data(valid_trials_subset, 3));
                            
                            % Update counters for data cleaning statistics
                            clean_trials = clean_trials + length(valid_trials_subset);
                            total_trials = total_trials + length(all_trials_subset);
                        else
                            MeanLook_location1{cond}(p, word, block) = NaN;
                        end
                    end
                end
            else
                % Not enough valid trials or missing word data
                for cond = 1:3
                    for word = 1:2
                        MeanLook_location1{cond}(p, word, block) = NaN;
                    end
                end
            end
        end
    end
end

%% Process data from Location 2 (learning scores)

fprintf('Processing Location 2 learning data...\n');

% Load raw looking time data
data_location2 = textread(fullfile(base_path, 'looktime/Location2_AllData.txt'));

% Get unique participant IDs
participants_location2 = unique(data_location2(:,1));
participants_location2_ID = unique(data_location2(:,1)) + 2100;

% Initialize arrays for demographic information
age_location2 = zeros(length(participants_location2_ID), 1);
sex_location2 = zeros(length(participants_location2_ID), 1);

% Initialize cell array for mean looking times
MeanLook_location2 = cell(1, 3);  % 3 conditions
for cond = 1:3
    MeanLook_location2{cond} = NaN(length(participants_location2_ID), 2, 3);  % participants × words × blocks
end

% Process each participant
for p = 1:length(participants_location2_ID)
    % Find valid trials (non-zero looking times)
    valid_trials = find(data_location2(:,1) == participants_location2(p) & data_location2(:,9) > 0);
    
    if ~isempty(valid_trials)
        % Extract demographic information
        age_location2(p) = data_location2(valid_trials(1), 2);
        sex_location2(p) = data_location2(valid_trials(1), 3);
        
        % Extract trial information
        trial_data = [data_location2(valid_trials, 6), data_location2(valid_trials, 8), ...
                     data_location2(valid_trials, 9), data_location2(valid_trials, 10), ... 
                     data_location2(valid_trials, 5)];
        
        % Calculate individual outlier threshold
        outlier_threshold = mean(trial_data(:,3)) + SD_cutoff * std(trial_data(:,3));
        
        % Process each block
        for block = 1:3
            % Find trials within outlier threshold for this block
            valid_block_trials = find(trial_data(:,3) < outlier_threshold & trial_data(:,5) == block);
            
            % Check if there are enough valid trials and both words are represented
            if length(valid_block_trials) > 1 && length(unique(trial_data(valid_block_trials, 2))) == 2
                % Process each condition
                for cond = 1:3
                    % Process each word
                    for word = 1:2
                        % Find valid trials for this condition, word, and block
                        valid_trials_subset = find(trial_data(:,1) == cond & trial_data(:,2) == word & ...
                                                  trial_data(:,3) < outlier_threshold & trial_data(:,5) == block);
                        
                        % Count all trials for this condition, word, and block (including outliers)
                        all_trials_subset = find(trial_data(:,1) == cond & trial_data(:,2) == word & trial_data(:,5) == block);
                        
                        % Calculate mean looking time if valid trials exist
                        if ~isempty(valid_trials_subset)
                            MeanLook_location2{cond}(p, word, block) = nanmean(trial_data(valid_trials_subset, 3));
                            
                            % Update counters for data cleaning statistics
                            clean_trials = clean_trials + length(valid_trials_subset);
                            total_trials = total_trials + length(all_trials_subset);
                        else
                            MeanLook_location2{cond}(p, word, block) = NaN;
                        end
                    end
                end
            else
                % Not enough valid trials or missing word data
                for cond = 1:3
                    for word = 1:2
                        MeanLook_location2{cond}(p, word, block) = NaN;
                    end
                end
            end
        end
    end
end

% Print data cleaning statistics
fprintf('Data cleaning ratio: %.2f%% (retained %d of %d trials)\n', ...
        100 * clean_trials / total_trials, clean_trials, total_trials);

%% Combine learning scores and calculate learning difference (word2 - word1)

fprintf('Combining learning data from both locations...\n');

% Initialize the combined data array
combined_data = [];

% Add Location 1 data
for i = 1:length(participants_location1_ID)
    for b = 1:3  % blocks
        for c = 1:3  % conditions
            % Calculate learning (word2 - word1) and average looking time
            learning_score = MeanLook_location1{c}(i, 2, b) - MeanLook_location1{c}(i, 1, b);
            avg_looking_time = (MeanLook_location1{c}(i, 2, b) + MeanLook_location1{c}(i, 1, b)) / 2;
            
            % Add row to combined data
            % Format: [location, ID, age, sex, block, condition, learning, avg_looking_time, NaN(for attention)]
            combined_data = [combined_data; [1, participants_location1_ID(i), age_location1(i), ...
                            sex_location1(i), b, c, learning_score, avg_looking_time, NaN]];
        end
    end
end

% Add Location 2 data
for i = 1:length(participants_location2_ID)
    for b = 1:3  % blocks
        for c = 1:3  % conditions
            % Calculate learning (word2 - word1) and average looking time
            learning_score = MeanLook_location2{c}(i, 2, b) - MeanLook_location2{c}(i, 1, b);
            avg_looking_time = (MeanLook_location2{c}(i, 2, b) + MeanLook_location2{c}(i, 1, b)) / 2;
            
            % Add row to combined data
            combined_data = [combined_data; [2, participants_location2_ID(i), age_location2(i), ...
                            sex_location2(i), b, c, learning_score, avg_looking_time, NaN]];
        end
    end
end

%% Process attention data from Location 1

fprintf('Processing Location 1 attention data...\n');

% Load attention data
attention_data_location1 = xlsread(fullfile(base_path, 'looktime/Location1_Summary.xlsx'));

% Handle potential ID adjustment (as seen in fs1 code)
if any(attention_data_location1(:,1) > 200)
    attention_data_location1(attention_data_location1(:,1) > 200, 1) = ...
        attention_data_location1(attention_data_location1(:,1) > 200, 1) - 100;
end

% Special case: adjust ID 6 to 101 as seen in fs1 code
attention_data_location1(attention_data_location1(:,1) == 6, 1) = 101;

% List of participant IDs (based on fs1 code)
location1_list = participants_location1;

% Initialize attention array
attention_scores = NaN(length(location1_list), 3, 3);  % participants × blocks × conditions

% Calculate mean attention scores for each participant, block, and condition
for block = 1:3
    for i = 1:length(location1_list)
        for cond = 1:3
            % Find matching trials
            matching_trials = find(attention_data_location1(:,1) == location1_list(i) & ...
                                  attention_data_location1(:,5) == cond & ...
                                  attention_data_location1(:,4) == block);
            
            % Calculate mean attention score if trials exist
            if ~isempty(matching_trials)
                attention_scores(i, block, cond) = nanmean(attention_data_location1(matching_trials, 8));
            end
        end
    end
end

% Adjust IDs to match the combined data format
location1_list_adjusted = location1_list + 1000;

% Add attention scores to Location 1 data in the combined dataset
for i = 1:size(combined_data, 1)
    if combined_data(i, 1) == 1  % Location 1
        % Find the participant index
        participant_idx = find(location1_list_adjusted == combined_data(i, 2));
        
        if ~isempty(participant_idx)
            block = combined_data(i, 5);
            condition = combined_data(i, 6);
            
            % Add attention score if available
            if ~isempty(attention_scores(participant_idx, block, condition)) && ...
               ~isnan(attention_scores(participant_idx, block, condition))
                combined_data(i, 9) = attention_scores(participant_idx, block, condition);
            end
        end
    end
end

%% Process attention data from Location 2

fprintf('Processing Location 2 attention data...\n');

% Load attention data
attention_data_location2 = xlsread(fullfile(base_path, 'looktime/Location2_Summary.xlsx'));

% ID adjustment for Location 2 (if needed)
if any(attention_data_location2(:,1) > 200)
    attention_data_location2(attention_data_location2(:,1) > 200, 1) = ...
        attention_data_location2(attention_data_location2(:,1) > 200, 1) - 100;
end

% Find unique combinations of participant, block, and condition
[group_combinations, ~, group_ids] = unique(attention_data_location2(:, [1, 7, 9]), 'rows', 'stable');

% Calculate attention proportions for each unique combination
attention_proportions = NaN(size(group_combinations, 1), 2);
for i = 1:size(group_combinations, 1)
    matching_rows = find(group_ids == i);
    
    total_time = 0;
    attention_time = 0;
    
    % Sum up times for each trial
    for j = 1:length(matching_rows)
        row_idx = matching_rows(j);
        
        if attention_data_location2(row_idx, 5) < 99
            total_time = total_time + attention_data_location2(row_idx, 6);
        else
            attention_time = attention_time + attention_data_location2(row_idx, 6);
        end
    end
    
    attention_proportions(i, :) = [total_time, attention_time];
end

% Combine group information with attention proportions
group_data = [group_combinations, attention_proportions];

% Adjust participant IDs to match the combined data format
group_data(:, 1) = group_data(:, 1) + 2000;

% Add attention proportions to Location 2 data in the combined dataset
for i = 1:size(combined_data, 1)
    if combined_data(i, 1) == 2  % Location 2
        % Find matching group data
        matching_row = find(group_data(:, 1) == combined_data(i, 2) & ...
                           group_data(:, 2) == combined_data(i, 5) & ...
                           group_data(:, 3) == combined_data(i, 6), 1, 'first');
        
        if ~isempty(matching_row)
            % Calculate and add attention proportion
            if group_data(matching_row, 4) > 0
                combined_data(i, 9) = group_data(matching_row, 5) / group_data(matching_row, 4);
            end
        end
    end
end

%% Identify complete data and check participant coverage

% Create a copy of the data for analysis
complete_data = combined_data;

% Identify rows with missing learning scores
missing_learning = zeros(size(complete_data, 1), 1);
for i = 1:size(complete_data, 1)
    if isnan(complete_data(i, 7))
        missing_learning(i) = 1;
    end
end

% Remove rows with missing learning scores
complete_data(missing_learning == 1, :) = [];

% Print the number of participants with complete learning data
fprintf('Participants with complete learning data: %d\n', length(unique(complete_data(:, 2))));

% Create another copy for attention analysis
attention_data = combined_data;

% Identify rows with missing attention scores
missing_attention = zeros(size(attention_data, 1), 1);
for i = 1:size(attention_data, 1)
    if isnan(attention_data(i, 9))
        missing_attention(i) = 1;
    end
end

% Remove rows with missing attention scores
attention_data(missing_attention == 1, :) = [];

% Print the number of participants with complete attention data
fprintf('Participants with complete attention data: %d\n', length(unique(attention_data(:, 2))));

%% Export combined data to Excel

fprintf('Exporting data to Excel...\n');

% Define column headers
column_headers = {'location', 'id', 'age', 'sex', 'block', 'condition', ...
                 'learning_score', 'total_looking_time', 'attention_proportion'};

% Define file path for export
output_file = fullfile(base_path, 'behavioral_data.xlsx');

% Check if file exists and delete if necessary
if exist(output_file, 'file') == 2
    delete(output_file);
end

% Try to write the data to Excel
try
    % Write headers and data
    xlswrite(output_file, column_headers, 'Sheet1', 'A1');
    xlswrite(output_file, combined_data, 'Sheet1', 'A2');
    fprintf('Data successfully exported to %s\n', output_file);
catch
    % Alternative file path in case of error
    alt_output_file = fullfile(base_path, 'behavioral_data_alt.xlsx');
    
    % Write headers and data to alternative location
    xlswrite(alt_output_file, column_headers, 'Sheet1', 'A1');
    xlswrite(alt_output_file, combined_data, 'Sheet1', 'A2');
    fprintf('Data exported to alternative location: %s\n', alt_output_file);
end

%% Summary statistics

fprintf('\nCalculating summary statistics...\n');

% Calculate overall means
mean_learning = nanmean(combined_data(:, 7));
mean_attention = nanmean(combined_data(:, 9));

fprintf('\nSummary Statistics:\n');
fprintf('Mean learning score: %.4f\n', mean_learning);
fprintf('Mean attention proportion: %.4f\n', mean_attention);

% Calculate means by location
location1_learning = nanmean(combined_data(combined_data(:, 1) == 1, 7));
location2_learning = nanmean(combined_data(combined_data(:, 1) == 2, 7));
location1_attention = nanmean(combined_data(combined_data(:, 1) == 1, 9));
location2_attention = nanmean(combined_data(combined_data(:, 1) == 2, 9));

fprintf('\nBy Location:\n');
fprintf('Location 1 mean learning: %.4f\n', location1_learning);
fprintf('Location 2 mean learning: %.4f\n', location2_learning);
fprintf('Location 1 mean attention: %.4f\n', location1_attention);
fprintf('Location 2 mean attention: %.4f\n', location2_attention);

% Calculate means by condition
c1 = combined_data(:, 6) == 1;  % Full gaze condition
c2 = combined_data(:, 6) == 2;  % Partial gaze condition  
c3 = combined_data(:, 6) == 3;  % No gaze condition

fprintf('\nBy Condition:\n');
fprintf('Full gaze learning: %.4f (n=%d)\n', nanmean(combined_data(c1, 7)), sum(~isnan(combined_data(c1, 7))));
fprintf('Partial gaze learning: %.4f (n=%d)\n', nanmean(combined_data(c2, 7)), sum(~isnan(combined_data(c2, 7))));
fprintf('No gaze learning: %.4f (n=%d)\n', nanmean(combined_data(c3, 7)), sum(~isnan(combined_data(c3, 7))));

fprintf('\nProcessing complete.\n');
