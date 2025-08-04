%% EEG Rejection Ratio Analysis Script
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Analyze EEG rejection ratios across different data categories
% and experimental conditions for infant EEG data
%
% This script analyzes preprocessed EEG data to calculate:
% 1. The proportion of unattended data
% 2. The proportion of rejected data after automated artifact rejection
% 3. The proportion of rejected data after manual artifact rejection
% 4. Comparison between datasets from two different locations
%
% Rejection codes: 777 = unattended, 999 = auto rejected, 888 = manually rejected

%% Load and preprocess data

% Set paths (modify as needed for your environment)
filepath_data1 = '/path/to/dataset1/';
filepath_data2 = '/path/to/dataset2/';

% Participant ID lists
% Dataset 1 participants
P_data1 = {'101','102','103','104','105','106','107','108','109','111','114','117','118','119','121','122','123','124','125','126','127','128','129','131','132','133','135'};

% Dataset 2 participants 
P_data2 = {'101','104','106','107','108','110','114','115','116','117','120','121','122','123','127'};

% Define electrodes to include (9-channel grid)
include = [4:6,15:17,26:28]'; 
chLabel = {'F3','Fz','F4','C3','Cz','C4','P3','Pz','P4'};

%% Load and analyze data from dataset 1
data1 = struct();
for p = 1:length(P_data1)
    % Load preprocessed EEG data
    filename = fullfile(filepath_data1, ['P', P_data1{p}, '*_AR.mat']);
    loadedData = load(filename, 'FamEEGart', 'StimEEGart');
    data1(p).FamEEGart = loadedData.FamEEGart;
    data1(p).StimEEGart = loadedData.StimEEGart;
end

%% Load and analyze data from dataset 2
data2 = struct();
for p = 1:length(P_data2)
    % Load preprocessed EEG data
    filename = fullfile(filepath_data2, ['P', P_data2{p}, '*_AR.mat']);
    loadedData = load(filename, 'FamEEGart', 'StimEEGart');
    data2(p).FamEEGart = loadedData.FamEEGart;
    data2(p).StimEEGart = loadedData.StimEEGart;
end

%% Calculate rejection ratios for both datasets
% Process dataset 1
eegcount_data1 = calculate_rejection_ratio(data1, include);

% Process dataset 2
eegcount_data2 = calculate_rejection_ratio(data2, include);

%% Calculate summary statistics for dataset 1
result_data1 = zeros(length(P_data1), 5);

% Sum across dimensions 2, 3, and 4 (blocks, conditions, phrases)
for i = 1:length(P_data1)
    for j = 1:5
        result_data1(i,j) = sum(sum(sum(eegcount_data1(i,:,:,:,j))));
    end
end

% Normalize by time units (200 samples)
result_data1 = result_data1/200;
total_data1 = sum(result_data1(:,:), 2);

%% Calculate summary statistics for dataset 2
result_data2 = zeros(length(P_data2), 5);

% Sum across dimensions 2, 3, and 4 (blocks, conditions, phrases)
for i = 1:length(P_data2)
    for j = 1:5
        result_data2(i,j) = sum(sum(sum(eegcount_data2(i,:,:,:,j))));
    end
end

% Normalize by time units
result_data2 = result_data2/200;
total_data2 = sum(result_data2(:,:), 2);

%% Combine results from both datasets for overall analysis
result_all = [result_data1; result_data2];
total_all = sum(result_all(:,:), 2);

%% Calculate and display statistics

% Display overall statistics
fprintf('\n==== OVERALL STATISTICS (BOTH DATASETS) ====\n');
fprintf('Total data per infant (mean, std): [%.2f, %.2f]\n', mean(total_all), std(total_all));

% Percentage of attended data
attended_percent = (total_all-result_all(:,2))./total_all;
fprintf('Attended data percentage (mean, std): [%.2f%%, %.2f%%]\n', 
    mean(attended_percent)*100, std(attended_percent)*100);

% Percentage of data after auto rejection
after_auto_percent = (total_all-result_all(:,2)-result_all(:,3))./total_all;
fprintf('Data remaining after auto rejection (mean, std): [%.2f%%, %.2f%%]\n', 
    mean(after_auto_percent)*100, std(after_auto_percent)*100);

% Percentage of data retained for final analysis
final_data_percent = (total_all-result_all(:,2)-result_all(:,3)-result_all(:,4))./total_all;
fprintf('Data retained for analysis (mean, std): [%.2f%%, %.2f%%]\n', 
    mean(final_data_percent)*100, std(final_data_percent)*100);

%% Compare unattended data between datasets
unattended_data1 = result_data1(:,2)./total_data1;
unattended_data2 = result_data2(:,2)./total_data2;

fprintf('\n==== DATASET COMPARISON: UNATTENDED DATA ====\n');
fprintf('Dataset 1 unattended (mean, std): [%.2f%%, %.2f%%]\n', 
    mean(unattended_data1)*100, std(unattended_data1)*100);
fprintf('Dataset 2 unattended (mean, std): [%.2f%%, %.2f%%]\n', 
    mean(unattended_data2)*100, std(unattended_data2)*100);

% Statistical test for unattended data between datasets
[h,p,~,stats] = ttest2(unattended_data1, unattended_data2);
fprintf('t-test for unattended data: t(%.0f) = %.2f, p = %.4f, significant: %s\n', 
    stats.df, stats.tstat, p, logical_to_text(h));

%% Compare auto-rejected data between datasets
auto_rejected_data1 = result_data1(:,3)./(total_data1-result_data1(:,2));
auto_rejected_data2 = result_data2(:,3)./(total_data2-result_data2(:,2));

fprintf('\n==== DATASET COMPARISON: AUTO-REJECTED DATA ====\n');
fprintf('Dataset 1 auto-rejected (mean, std): [%.2f%%, %.2f%%]\n', 
    mean(auto_rejected_data1)*100, std(auto_rejected_data1)*100);
fprintf('Dataset 2 auto-rejected (mean, std): [%.2f%%, %.2f%%]\n', 
    mean(auto_rejected_data2)*100, std(auto_rejected_data2)*100);

% Statistical test for auto-rejected data between datasets
[h,p,~,stats] = ttest2(auto_rejected_data1, auto_rejected_data2);
fprintf('t-test for auto-rejected data: t(%.0f) = %.2f, p = %.4f, significant: %s\n', 
    stats.df, stats.tstat, p, logical_to_text(h));

%% Condition comparison analysis for dataset 2
analyze_conditions(eegcount_data2);

%% Helper Functions

function eegcount = calculate_rejection_ratio(data, include)
    % Calculate rejection ratios for EEG data
    % Returns a matrix of dimensions: participants × blocks × conditions × phrases × categories
    % where categories are: 1=valid, 2=unattended, 3=auto-rejected, 4=manually-rejected, 5=NaN
    
    [num_participants, ~] = size(data);
    eegcount = zeros(num_participants, 3, 3, 4, 5); % Assuming 3 blocks, 3 conditions, 4 phrases
    
    for p = 1:num_participants
        FamEEGart = data(p).FamEEGart;
        StimEEGart = data(p).StimEEGart;
        
        % Process EEG data
        iEEG = cell(1, size(FamEEGart, 1));
        aEEG = cell(1, size(FamEEGart, 1));
        
        % Extract selected channels
        for block = 1:size(FamEEGart, 1)
            if ~isempty(FamEEGart{block})
                for cond = 1:size(FamEEGart{block}, 1)
                    for phrase = 1:size(FamEEGart{block}, 2)
                        if ~isempty(FamEEGart{block}{cond,phrase}) && size(FamEEGart{block}{cond,phrase}, 1) > 1
                            for chan = 1:length(include)
                                iEEG{block}{cond,phrase}(:,chan) = FamEEGart{block}{cond,phrase}(:,include(chan));
                                aEEG{block}{cond,phrase}(:,chan) = StimEEGart{block}{cond,phrase}(:,include(chan));
                            end
                        end
                    end
                end
            else
                iEEG{block} = [];
                aEEG{block} = [];
            end
        end
        
        % Count valid and rejected time points
        for block = 1:size(FamEEGart, 1)
            if ~isempty(iEEG{block})
                for cond = 1:size(FamEEGart{block}, 1)
                    for phrase = 1:size(FamEEGart{block}, 2)
                        if ~isempty(iEEG{block}{cond,phrase})
                            temp = iEEG{block}{cond,phrase};
                            eegcount(p,block,cond,phrase,1) = length(find(abs(temp) < 700));
                            eegcount(p,block,cond,phrase,2) = length(find(temp == 777));  % Unattended
                            eegcount(p,block,cond,phrase,3) = length(find(temp == 999));  % Auto-rejected
                            eegcount(p,block,cond,phrase,4) = length(find(temp == 888));  % Manually rejected
                            eegcount(p,block,cond,phrase,5) = length(find(isnan(temp)));  % NaN values
                        end
                    end
                end
            end
        end
    end
end

function analyze_conditions(data)
    % Analyze rejection ratios across different experimental conditions
    % Performs ANOVA to test for significant differences
    
    % Extract data for conditions 1, 2, and 3
    conditions = [1, 2, 3];
    num_conditions = length(conditions);
    
    % Get matrix dimensions
    [num_participants, num_blocks, ~, num_phrases, ~] = size(data);
    
    % Initialize array to store rejection ratios
    rejection_ratios = zeros(num_participants, num_conditions);
    
    % Calculate rejection ratios for each condition
    for c_idx = 1:num_conditions
        cond = conditions(c_idx);
        
        for p_idx = 1:num_participants
            total_time = 0;
            rejected_time = 0;
            
            % Sum across all blocks and phrases
            for b = 1:num_blocks
                for ph = 1:num_phrases
                    % Total time is sum of categories 1-5
                    current_total = sum(data(p_idx, b, cond, ph, 1:5));
                    total_time = total_time + current_total;
                    
                    % Rejected time is sum of categories 2-4
                    current_rejected = sum(data(p_idx, b, cond, ph, 2:4));
                    rejected_time = rejected_time + current_rejected;
                end
            end
            
            % Calculate rejection ratio (proportion of time rejected)
            if total_time > 0
                rejection_ratios(p_idx, c_idx) = rejected_time / total_time;
            else
                rejection_ratios(p_idx, c_idx) = NaN; % Handle division by zero
            end
        end
    end
    
    % Prepare data for ANOVA
    [p, ~] = size(rejection_ratios);
    anova_data = reshape(rejection_ratios, [], 1);
    group_labels = repmat(1:num_conditions, p, 1);
    group_labels = reshape(group_labels, [], 1);
    
    % Remove NaN values if any
    valid_idx = ~isnan(anova_data);
    anova_data = anova_data(valid_idx);
    group_labels = group_labels(valid_idx);
    
    % Convert group labels to categorical variable
    group_categories = categorical(group_labels, 1:num_conditions, {'Condition 1', 'Condition 2', 'Condition 3'});
    
    % Perform one-way ANOVA
    [p_value, tbl, ~] = anova1(anova_data, group_categories, 'off');
    
    % Display results
    fprintf('\n==== CONDITION COMPARISON (ANOVA) ====\n');
    disp(tbl);
    fprintf('ANOVA p-value: %.4f, significant: %s\n', p_value, logical_to_text(p_value < 0.05));
    
    % Calculate descriptive statistics for each condition
    condition_means = zeros(1, num_conditions);
    condition_stds = zeros(1, num_conditions);
    
    for c = 1:num_conditions
        condition_data = rejection_ratios(:, c);
        condition_means(c) = mean(condition_data, 'omitnan');
        condition_stds(c) = std(condition_data, 'omitnan');
    end
    
    % Display descriptive statistics
    fprintf('\nDescriptive statistics for rejection ratios:\n');
    for c = 1:num_conditions
        fprintf('Condition %d: Mean = %.2f%%, SD = %.2f%%\n', 
            c, condition_means(c)*100, condition_stds(c)*100);
    end
end

function result = logical_to_text(logical_value)
    % Convert logical value to yes/no text
    if logical_value
        result = 'Yes';
    else
        result = 'No';
    end
end
