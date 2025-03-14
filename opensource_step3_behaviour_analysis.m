%% Statistical Analysis Script for CDI and Learning Outcomes
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.
%
% Purpose: Analyze relationships between language learning outcomes, gaze conditions, 
% attentional measures, and CDI gesture scores
%
% This script:
% 1. Loads behavioral data (looking time, attention, CDI scores)
% 2. Merges data from multiple sources
% 3. Conducts statistical analyses (t-tests, linear mixed models)
% 4. Produces statistical outputs for tables and figures

%% Load and prepare data

% Set base path
base_path = '/path/to/data/';

% Load CDI (Communicative Development Inventory) data
cdi_data = xlsread(fullfile(base_path, 'CDI', 'CDI_data.xlsx'), 'Sheet1');
cdi_details = xlsread(fullfile(base_path, 'CDI', 'CDI_data.xlsx'), 'cdi');

% Map CDI scores to participant IDs
for i = 1:size(cdi_data, 1)
    id = cdi_data(i, 2);
    for j = 1:size(cdi_details, 1)
        if id == cdi_details(j, 1)
            % Extract CDI Gesture score
            cdi_data(i, 12) = cdi_details(j, 4);
            break
        end
    end
end

% Load behavioral data
behavioral_data = xlsread(fullfile(base_path, 'table', 'behavioral_data.xlsx'));

% Extract key variables
Country = categorical(behavioral_data(:, 1));    % Location (1 = Location 1, 2 = Location 2)
ID = categorical(behavioral_data(:, 2));         % Participant ID
AGE = behavioral_data(:, 3);                     % Age in months
SEX = categorical(behavioral_data(:, 4));        % Sex
learning = behavioral_data(:, 7);                % Learning score (word2 - word1 looking time)
Attention = behavioral_data(:, 9) * 60;          % Attention score (converted to seconds)
blocks = behavioral_data(:, 5);                  % Experimental block

% Create condition indices
c1 = find(behavioral_data(:, 6) == 1);  % Full gaze condition
c2 = find(behavioral_data(:, 6) == 2);  % Partial gaze condition
c3 = find(behavioral_data(:, 6) == 3);  % No gaze condition

% Create cohort indices
location1 = find(behavioral_data(:, 1) == 1);  % Location 1 cohort
location2 = find(behavioral_data(:, 1) == 2);  % Location 2 cohort

%% Load and merge gaze data

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
            CDIG(i) = cdi_data(j, 12);
            break;
        end
    end
end

%% Table 1: Performance metrics by condition

% Calculate descriptive statistics for learning
N1 = sum(~isnan(learning(c1)));
N2 = sum(~isnan(learning(c2)));
N3 = sum(~isnan(learning(c3)));

learning_stats = [nanmean(learning(c1)), nanstd(learning(c1)), ...
                  nanmean(learning(c2)), nanstd(learning(c2)), ...
                  nanmean(learning(c3)), nanstd(learning(c3))];

fprintf('\nTable 1: Learning scores by condition\n');
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

%% Learning analysis with covariates

% Adjust learning scores for covariates (age, sex, location)
% Full gaze condition
X1 = [ones(size(behavioral_data(c1,1))), behavioral_data(c1,[1,3,4])];
[~,~,resid1] = regress(learning(c1), X1);
adjusted_scores1 = resid1 + nanmean(learning(c1));

% Partial gaze condition
X2 = [ones(size(behavioral_data(c2,1))), behavioral_data(c2,[1,3,4])];
[~,~,resid2] = regress(learning(c2), X2);
adjusted_scores2 = resid2 + nanmean(learning(c2));

% No gaze condition
X3 = [ones(size(behavioral_data(c3,1))), behavioral_data(c3,[1,3,4])];
[~,~,resid3] = regress(learning(c3), X3);
adjusted_scores3 = resid3 + nanmean(learning(c3));

% One-sample t-tests on adjusted scores
[h1, p1, CI1, stats1] = ttest(adjusted_scores1);
[h2, p2, CI2, stats2] = ttest(adjusted_scores2);
[h3, p3, CI3, stats3] = ttest(adjusted_scores3);

% Apply FDR correction
q = mafdr([p1, p2, p3], 'BHFDR', true);

fprintf('\nOne-sample t-tests on learning scores (with covariates)\n');
fprintf('Full gaze: t(%d) = %.2f, p = %.4f, corrected p = %.4f\n', stats1.df, stats1.tstat, p1, q(1));
fprintf('Partial gaze: t(%d) = %.2f, p = %.4f, corrected p = %.4f\n', stats2.df, stats2.tstat, p2, q(2));
fprintf('No gaze: t(%d) = %.2f, p = %.4f, corrected p = %.4f\n', stats3.df, stats3.tstat, p3, q(3));

%% Observed power analysis

% Calculate Cohen's d and observed power for significant effect
cohen_d = stats1.tstat / sqrt(stats1.df + 1);
n = stats1.df + 1;
alpha = 0.05;

% Calculate non-centrality parameter
ncp = cohen_d * sqrt(n);

% Calculate observed power
df = n - 1;
crit_t = tinv(1-alpha/2, df);  % Critical t-value (two-tailed)
observed_power = 1 - nctcdf(crit_t, df, ncp);

fprintf('\nObserved power analysis (Full gaze condition)\n');
fprintf('Cohen\'s d: %.4f\n', cohen_d);
fprintf('Observed power: %.4f\n', observed_power);

%% Create data table for mixed models

% Create categorical variable for condition
cond = categorical(behavioral_data(:, 6));

% Create combined data table
dataTable = table(cond, duration, onsetnum, blocks, Country, ID, AGE, SEX, CDIG, learning, Attention);

%% Attention analysis by location

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

%% CDI and learning analyses

% Mixed model for overall learning with CDI Gesture scores
fprintf('\nMixed model: Learning and CDI Gesture scores (all conditions)\n');
lme_cdig = fitlme(dataTable, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig);

% Separate analyses by condition
fprintf('\nMixed model: Learning and CDI Gesture scores (Full gaze condition)\n');
dataTable_c1 = dataTable(c1, :);
lme_cdig_c1 = fitlme(dataTable_c1, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig_c1);

fprintf('\nMixed model: Learning and CDI Gesture scores (Partial gaze condition)\n');
dataTable_c2 = dataTable(c2, :);
lme_cdig_c2 = fitlme(dataTable_c2, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig_c2);

fprintf('\nMixed model: Learning and CDI Gesture scores (No gaze condition)\n');
dataTable_c3 = dataTable(c3, :);
lme_cdig_c3 = fitlme(dataTable_c3, 'learning ~ CDIG + AGE + SEX + Country + (1|ID)');
disp(lme_cdig_c3);

%% CDI comparison between locations

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