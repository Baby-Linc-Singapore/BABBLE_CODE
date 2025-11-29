%% Statistical Analysis Script for Behavioral Data
% NOTE: This code demonstrates the analytical methodology.
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to
% NTU's open access policy.
%
% Purpose: Analyze attention measures, CDI scores, and their relationships
%
% NOTE: Learning effect analysis (three-tier hierarchical testing) is in a separate script:
%   → See opensource_step03a_learning_three_tier_analysis.m
%
% This script analyzes:
% - Attention measures across gaze conditions and cohorts
% - CDI gesture scores
% - Correlations between attention and learning
% - Country (cohort) effects
%
% This script:
% 1. Loads behavioral data (looking time, attention, CDI scores)
% 2. Merges data from multiple sources
% 3. Conducts statistical analyses (t-tests, linear mixed models)
% 4. Produces statistical outputs for tables and figures

%% Initialize environment
clear all
clc

% Set base path (modify as needed)
base_path = '/path/to/data/';

%% Load CDI (Communicative Development Inventory) data

fprintf('Loading CDI data...\n');

% Load CDI data and details
cdi_data = xlsread(fullfile(base_path, 'CDI', 'CDI_questionnaire_data.xlsx'), 'Sheet1');
cdi_details = xlsread(fullfile(base_path, 'CDI', 'CDI_questionnaire_data.xlsx'), 'cdi');

% Map CDI scores to participant IDs
for i = 1:size(cdi_data, 1)
    id = cdi_data(i, 2);
    for j = 1:size(cdi_details, 1)
        if id == cdi_details(j, 1)
            % Extract CDI scores:  Gestures (G)
            cdi_data(i, 12) = cdi_details(j, 4);
            break
        end
    end
end

%% Load behavioral data

fprintf('Loading behavioral data...\n');

% Load behavioral data
behavioral_data = xlsread(fullfile(base_path, 'table', 'behavioral_data.xlsx'));

% Extract key variables
Country = categorical(behavioral_data(:, 1));    % Location (1 = Location 1, 2 = Location 2)
ID = categorical(behavioral_data(:, 2));         % Participant ID
AGE = behavioral_data(:, 3);                     % Age in months
SEX = categorical(behavioral_data(:, 4));        % Sex (1 = male, 2 = female)
blocks = categorical(behavioral_data(:, 5));     % Experimental block
learning = behavioral_data(:, 7);                % Learning score (word2 - word1 looking time)
total_looking_time = behavioral_data(:, 8);      % Total looking time (word1 + word2)
Attention = behavioral_data(:, 9) * 60;          % Attention score (converted to seconds)

% Create condition indices
c1 = find(behavioral_data(:, 6) == 1);  % Full gaze condition
c2 = find(behavioral_data(:, 6) == 2);  % Partial gaze condition
c3 = find(behavioral_data(:, 6) == 3);  % No gaze condition

% Create cohort indices
location1 = find(behavioral_data(:, 1) == 1);  % Location 1 cohort
location2 = find(behavioral_data(:, 1) == 2);  % Location 2 cohort

%% Load and merge gaze data

fprintf('Loading gaze data...\n');

% Load gaze data for Location 1
load(fullfile(base_path, 'attendance_data_location1.mat'));
location1_onset_duration = onset_duration;
location1_onset_number = onset_number;
location1_participant_list = participant_list;

% Load gaze data for Location 2
load(fullfile(base_path, 'attendance_data_location2.mat'));
location2_onset_duration = onset_duration;
location2_onset_number = onset_number;
location2_participant_list = participant_list;

% Initialize arrays for gaze metrics
duration = NaN(size(behavioral_data, 1), 1);
onsetnum = NaN(size(behavioral_data, 1), 1);

% Merge Location 1 gaze data
for i = 1:size(behavioral_data, 1)
    if behavioral_data(i, 1) == 1  % Location 1
        id = behavioral_data(i, 2) - 1000;  % Convert to original ID format
        block = behavioral_data(i, 5);
        condition = behavioral_data(i, 6);
        
        % Find matching participant in gaze data
        for j = 1:length(location1_participant_list)
            if str2double(location1_participant_list{j}) == id
                % Extract gaze duration
                temp_duration = squeeze(location1_onset_duration(j, condition, block, :));
                temp_data = [];
                
                % Combine data across phrases
                for k = 1:length(temp_duration)
                    if ~isempty(temp_duration{k})
                        temp_data = [temp_data; temp_duration{k}];
                    end
                end
                
                % Calculate mean duration if data exists
                if ~isempty(temp_data)
                    duration(i) = mean(temp_data);
                end
                
                % Extract gaze onset count
                temp_onset = squeeze(location1_onset_number(j, condition, block, :));
                temp_data = [];
                
                % Combine data across phrases
                for k = 1:length(temp_onset)
                    if ~isempty(temp_onset{k})
                        temp_data = [temp_data; temp_onset{k}];
                    end
                end
                
                % Calculate total onsets if data exists
                if ~isempty(temp_data)
                    onsetnum(i) = sum(temp_data);
                end
                
                break;
            end
        end
    end
end

% Merge Location 2 gaze data
for i = 1:size(behavioral_data, 1)
    if behavioral_data(i, 1) == 2  % Location 2
        id = behavioral_data(i, 2) - 2000;  % Convert to original ID format
        block = behavioral_data(i, 5);
        condition = behavioral_data(i, 6);
        
        % Find matching participant in gaze data
        for j = 1:length(location2_participant_list)
            if str2double(location2_participant_list{j}) == id
                % Extract gaze duration
                temp_duration = squeeze(location2_onset_duration(j, condition, block, :));
                temp_data = [];
                
                % Combine data across phrases
                for k = 1:length(temp_duration)
                    if ~isempty(temp_duration{k})
                        temp_data = [temp_data; temp_duration{k}];
                    end
                end
                
                % Calculate mean duration if data exists
                if ~isempty(temp_data)
                    duration(i) = mean(temp_data);
                end
                
                % Extract gaze onset count
                temp_onset = squeeze(location2_onset_number(j, condition, block, :));
                temp_data = [];
                
                % Combine data across phrases
                for k = 1:length(temp_onset)
                    if ~isempty(temp_onset{k})
                        temp_data = [temp_data; temp_onset{k}];
                    end
                end
                
                % Calculate total onsets if data exists
                if ~isempty(temp_data)
                    onsetnum(i) = sum(temp_data);
                end
                
                break;
            end
        end
    end
end

% Convert duration from samples to seconds
duration = duration / 200;  % Sampling rate = 200 Hz

% Extract CDI Gesture scores
CDIG = NaN(size(behavioral_data, 1), 1);

for i = 1:size(behavioral_data, 1)
    for j = 1:size(cdi_data, 1)
        if behavioral_data(i, 2) == cdi_data(j, 2)
            CDIG(i) = cdi_data(j, 12);  % CDI Gestures
            break;
        end
    end
end

%% Table 1: Performance metrics by condition

fprintf('\n=== TABLE 1: PERFORMANCE METRICS BY CONDITION ===\n');

% Calculate descriptive statistics for learning
N1 = sum(~isnan(learning(c1)));
N2 = sum(~isnan(learning(c2)));
N3 = sum(~isnan(learning(c3)));

learning_stats = [nanmean(learning(c1)), nanstd(learning(c1)), ...
                  nanmean(learning(c2)), nanstd(learning(c2)), ...
                  nanmean(learning(c3)), nanstd(learning(c3))];

fprintf('\nLearning scores by condition\n');
fprintf('Full gaze: %.2f ± %.2f (n=%d)\n', learning_stats(1), learning_stats(2), N1);
fprintf('Partial gaze: %.2f ± %.2f (n=%d)\n', learning_stats(3), learning_stats(4), N2);
fprintf('No gaze: %.2f ± %.2f (n=%d)\n', learning_stats(5), learning_stats(6), N3);

% Calculate descriptive statistics for attention
N1 = sum(~isnan(Attention(c1)));
N2 = sum(~isnan(Attention(c2)));
N3 = sum(~isnan(Attention(c3)));

attention_stats = [nanmean(Attention(c1)), nanstd(Attention(c1)), ...
                   nanmean(Attention(c2)), nanstd(Attention(c2)), ...
                   nanmean(Attention(c3)), nanstd(Attention(c3))];

fprintf('\nAttention duration by condition (seconds)\n');
fprintf('Full gaze: %.2f ± %.2f (n=%d)\n', attention_stats(1), attention_stats(2), N1);
fprintf('Partial gaze: %.2f ± %.2f (n=%d)\n', attention_stats(3), attention_stats(4), N2);
fprintf('No gaze: %.2f ± %.2f (n=%d)\n', attention_stats(5), attention_stats(6), N3);

%% NOTE: Learning Effect Analysis
% Learning analysis (three-tier hierarchical testing) has been moved to:
%   → opensource_step03a_learning_three_tier_analysis.m
%
% That script performs:
%   - Tier 1: Omnibus ANOVA testing condition differences
%   - Tier 2: Pairwise contrasts (FDR-corrected)
%   - Tier 3: Within-condition baseline tests (learning > 0)

%% Create data table for mixed models

% Create categorical variable for condition
cond = categorical(behavioral_data(:, 6));

% Create combined data table
dataTable = table(cond, duration, onsetnum, blocks, Country, ID, AGE, SEX, CDIG, learning, Attention);

%% Attention analysis by location

fprintf('\n=== ATTENTION ANALYSIS BY LOCATION ===\n');

% Mixed model for number of attention onsets by location
fprintf('\nMixed model: Number of attention onsets by location\n');
lme_onsetnum = fitlme(dataTable, 'onsetnum ~ AGE + SEX + Country + (1|ID)');
disp(lme_onsetnum);

% Mixed model for attention duration by location
fprintf('\nMixed model: Attention duration by location\n');
lme_duration = fitlme(dataTable, 'duration ~ AGE + SEX + Country + (1|ID)');
disp(lme_duration);

% Mixed model for total attention by location
fprintf('\nMixed model: Total attention by location\n');
lme_attention = fitlme(dataTable, 'Attention ~ AGE + SEX + Country + (1|ID)');
disp(lme_attention);

%% Attention analysis by condition

fprintf('\n=== ATTENTION ANALYSIS BY CONDITION ===\n');

% Mixed model for number of attention onsets by condition
fprintf('\nMixed model: Number of attention onsets by condition\n');
lme_onsetnum_cond = fitlme(dataTable, 'onsetnum ~ AGE + SEX + Country + cond + (1|ID)');
disp(lme_onsetnum_cond);

% Mixed model for attention duration by condition
fprintf('\nMixed model: Attention duration by condition\n');
lme_duration_cond = fitlme(dataTable, 'duration ~ AGE + SEX + Country + cond + (1|ID)');
disp(lme_duration_cond);

% Mixed model for total attention by condition
fprintf('\nMixed model: Total attention by condition\n');
lme_attention_cond = fitlme(dataTable, 'Attention ~ AGE + SEX + Country + cond + (1|ID)');
disp(lme_attention_cond);

% Pairwise comparisons (swap conditions to test Partial vs No gaze)
fprintf('\nPairwise comparison: Partial vs No gaze\n');
temp_cond = behavioral_data(:, 6);
temp_cond(temp_cond == 1) = 4;  % Temporarily change Full gaze
temp_cond(temp_cond == 2) = 1;  % Partial becomes reference
temp_cond(temp_cond == 4) = 2;  % Full becomes second category
cond_swapped = categorical(temp_cond);

dataTable_swapped = table(cond_swapped, duration, onsetnum, blocks, Country, ID, AGE, SEX, CDIG, learning, Attention);
lme_attention_swap = fitlme(dataTable_swapped, 'Attention ~ AGE + SEX + Country + cond_swapped + (1|ID)');
disp(lme_attention_swap);

%% Learning-attention relationship

fprintf('\n=== LEARNING-ATTENTION RELATIONSHIP ===\n');

% Overall correlation across all conditions
fprintf('\nMixed model: Learning and attention (all conditions)\n');
lme_learning_attention = fitlme(dataTable, 'learning ~ Attention + AGE + SEX + Country + (1|ID)');
disp(lme_learning_attention);

% Partial correlation
valid = intersect(find(~isnan(Attention)), find(~isnan(learning)));
[r, p] = partialcorr(Attention(valid), learning(valid), behavioral_data(valid, [1,3,4]));
fprintf('Partial correlation (controlling for location, age, sex): r = %.3f, p = %.4f\n', r, p);

% Separate analyses by condition
fprintf('\nMixed model: Learning and attention (Full gaze condition)\n');
dataTable_c1 = dataTable(c1, :);
lme_learning_attention_c1 = fitlme(dataTable_c1, 'learning ~ Attention + AGE + SEX + Country + (1|ID)');
disp(lme_learning_attention_c1);

fprintf('\nMixed model: Learning and attention (Partial gaze condition)\n');
dataTable_c2 = dataTable(c2, :);
lme_learning_attention_c2 = fitlme(dataTable_c2, 'learning ~ Attention + AGE + SEX + Country + (1|ID)');
disp(lme_learning_attention_c2);

fprintf('\nMixed model: Learning and attention (No gaze condition)\n');
dataTable_c3 = dataTable(c3, :);
lme_learning_attention_c3 = fitlme(dataTable_c3, 'learning ~ Attention + AGE + SEX + Country + (1|ID)');
disp(lme_learning_attention_c3);

%% Learning by location

fprintf('\n=== LEARNING BY LOCATION ===\n');

% Mixed model for learning by location
fprintf('\nMixed model: Learning by location\n');
lme_learning_location = fitlme(dataTable, 'learning ~ AGE + SEX + Country + (1|ID)');
disp(lme_learning_location);

%% CDI and learning analyses

fprintf('\n=== CDI AND LEARNING ANALYSES ===\n');

% Mixed model for overall learning with CDI Gesture scores
fprintf('\nMixed model: Learning and CDI Gesture scores (all conditions)\n');
lme_cdig = fitlme(dataTable, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig);

% Separate analyses by condition
fprintf('\nMixed model: Learning and CDI Gesture scores (Full gaze condition)\n');
lme_cdig_c1 = fitlme(dataTable_c1, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig_c1);

fprintf('\nMixed model: Learning and CDI Gesture scores (Partial gaze condition)\n');
lme_cdig_c2 = fitlme(dataTable_c2, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig_c2);

fprintf('\nMixed model: Learning and CDI Gesture scores (No gaze condition)\n');
lme_cdig_c3 = fitlme(dataTable_c3, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig_c3);

% Collect p-values for FDR correction
cdi_p_values = [lme_cdig.Coefficients.pValue(2), lme_cdig_c1.Coefficients.pValue(2), ...
                lme_cdig_c2.Coefficients.pValue(2), lme_cdig_c3.Coefficients.pValue(2)];
cdi_q_values = mafdr(cdi_p_values, 'BHFDR', true);

fprintf('\nFDR-corrected p-values for CDI-learning relationships:\n');
fprintf('All conditions: p = %.4f, corrected p = %.4f\n', cdi_p_values(1), cdi_q_values(1));
fprintf('Full gaze: p = %.4f, corrected p = %.4f\n', cdi_p_values(2), cdi_q_values(2));
fprintf('Partial gaze: p = %.4f, corrected p = %.4f\n', cdi_p_values(3), cdi_q_values(3));
fprintf('No gaze: p = %.4f, corrected p = %.4f\n', cdi_p_values(4), cdi_q_values(4));

%% CDI comparison between locations

fprintf('\n=== CDI COMPARISON BETWEEN LOCATIONS ===\n');

% Get unique participants for demographic analysis
[~, unique_indices] = unique(behavioral_data(:, 2));
unique_data = dataTable(unique_indices, :);

% Compare CDI Gesture scores between locations
fprintf('\nMixed model: CDI Gesture scores by location\n');
lme_cdig_location = fitlme(unique_data, 'CDIG ~ AGE + SEX + Country');
disp(lme_cdig_location);

% Extract CDI Gesture scores by location for t-test
location1_cdig = CDIG(behavioral_data(:, 1) == 1);
location1_cdig = location1_cdig(~isnan(location1_cdig));
location1_cdig = unique(location1_cdig);  % Get unique values per participant

location2_cdig = CDIG(behavioral_data(:, 1) == 2);
location2_cdig = location2_cdig(~isnan(location2_cdig));
location2_cdig = unique(location2_cdig);  % Get unique values per participant

% Perform t-test comparing CDI Gesture scores between locations
[h, p, ~, stats] = ttest2(location1_cdig, location2_cdig);
fprintf('\nCDI Gesture score comparison between locations\n');
fprintf('Location 1: %.2f ± %.2f\n', mean(location1_cdig), std(location1_cdig));
fprintf('Location 2: %.2f ± %.2f\n', mean(location2_cdig), std(location2_cdig));
fprintf('t(%d) = %.2f, p = %.4f\n', stats.df, stats.tstat, p);

%% Prepare data for plotting

fprintf('\n=== DATA FOR PLOTTING ===\n');

% Word 1 and Word 2 looking times (for Figure 1d)
w = (behavioral_data(:, 8) * 2 - behavioral_data(:, 7)) ./ 2;  % Word 1
nw = (behavioral_data(:, 8) * 2 + behavioral_data(:, 7)) ./ 2;  % Word 2

fprintf('\nLooking time data for Figure 1d\n');
fprintf('Full gaze - Word 1: %.2f ± %.2f\n', nanmean(w(c1)), nanstd(w(c1)));
fprintf('Full gaze - Word 2: %.2f ± %.2f\n', nanmean(nw(c1)), nanstd(nw(c1)));
fprintf('Partial gaze - Word 1: %.2f ± %.2f\n', nanmean(w(c2)), nanstd(w(c2)));
fprintf('Partial gaze - Word 2: %.2f ± %.2f\n', nanmean(nw(c2)), nanstd(nw(c2)));
fprintf('No gaze - Word 1: %.2f ± %.2f\n', nanmean(w(c3)), nanstd(w(c3)));
fprintf('No gaze - Word 2: %.2f ± %.2f\n', nanmean(nw(c3)), nanstd(nw(c3)));

% Data for Figure 2 and Supplementary Figure S2
fprintf('\nAttention data by location (for Figure 2)\n');
fprintf('Location 1 onset number: %.2f ± %.2f\n', nanmean(onsetnum(location1)), nanstd(onsetnum(location1)));
fprintf('Location 2 onset number: %.2f ± %.2f\n', nanmean(onsetnum(location2)), nanstd(onsetnum(location2)));
fprintf('Location 1 duration: %.2f ± %.2f\n', nanmean(duration(location1)), nanstd(duration(location1)));
fprintf('Location 2 duration: %.2f ± %.2f\n', nanmean(duration(location2)), nanstd(duration(location2)));
fprintf('Location 1 attention: %.2f ± %.2f\n', nanmean(Attention(location1)), nanstd(Attention(location1)));
fprintf('Location 2 attention: %.2f ± %.2f\n', nanmean(Attention(location2)), nanstd(Attention(location2)));

fprintf('\nAttention data by condition (for Figure 2)\n');
fprintf('Full gaze onset number: %.2f ± %.2f\n', nanmean(onsetnum(c1)), nanstd(onsetnum(c1)));
fprintf('Partial gaze onset number: %.2f ± %.2f\n', nanmean(onsetnum(c2)), nanstd(onsetnum(c2)));
fprintf('No gaze onset number: %.2f ± %.2f\n', nanmean(onsetnum(c3)), nanstd(onsetnum(c3)));
fprintf('Full gaze duration: %.2f ± %.2f\n', nanmean(duration(c1)), nanstd(duration(c1)));
fprintf('Partial gaze duration: %.2f ± %.2f\n', nanmean(duration(c2)), nanstd(duration(c2)));
fprintf('No gaze duration: %.2f ± %.2f\n', nanmean(duration(c3)), nanstd(duration(c3)));

fprintf('\nAnalysis complete.\n');
