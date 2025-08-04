%% Statistical Analysis for EEG Connectivity
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Perform statistical analyses to identify significant connectivity patterns
% in EEG data compared to surrogate data, and examine effects of experimental conditions
%
% This script:
% 1. Loads real and surrogate connectivity data
% 2. Compares connectivity strength against surrogate distributions
% 3. Applies False Discovery Rate (FDR) correction for multiple comparisons
% 4. Tests effects of experimental conditions on connectivity patterns
% 5. Creates visualizations for significant connectivity patterns

%% Initialize environment
clc;
clear all;

% Set base path and load data
base_path = '/path/to/data/';
connectivity_type = 'GPDC';  % PDC or GPDC

% Load real and surrogate data
load(fullfile(base_path, ['data_read_surr_', connectivity_type, '2.mat']), 'data_surr', 'data');

% Optional: Remove outliers
cutlist = [];  % Indices of participants to exclude
data(cutlist, :) = [];

%% Define analysis parameters

% Extract experimental groups
g1 = find(data(:, 6) == 1);  % Full gaze condition
g2 = find(data(:, 6) == 2);  % Partial gaze condition
g3 = find(data(:, 6) == 3);  % No gaze condition

% Extract variables
AGE = data(:, 3);
SEX = categorical(data(:, 4));
COUNTRY = categorical(data(:, 1));
blocks = categorical(data(:, 5));
CONDGROUP = categorical(data(:, 6));
learning = data(:, 7);
atten = data(:, 9);
ID = categorical(data(:, 2));

% Setup naming for connectivity matrices
nodes = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
frequencyBands = {'Delta', 'Theta', 'Alpha'};
connectionTypes = {'II', 'AA', 'AI', 'IA'};

% Generate connectivity labels
connectionTitles = {};
for conn = 1:length(connectionTypes)
    for freq = 1:length(frequencyBands)
        for src = 1:length(nodes)
            for dest = 1:length(nodes)
                connectionTitles{end+1} = sprintf('%s_%s_%s_to_%s', ...
                                                 connectionTypes{conn}, frequencyBands{freq}, ...
                                                 nodes{src}, nodes{dest});
            end
        end
    end
end

% Combine all titles
titles = {'Country', 'ID', 'Age', 'Sex', 'Block', 'Condition', 'Learning', ...
         'Attention', connectionTitles{:}};

% Define indices for connectivity matrices
% Each matrix has 81 elements (9x9 grid)
ii1 = [10:9+81];       % Infant-to-Infant Delta
ii2 = [10+81*1:9+81*2];  % Infant-to-Infant Theta
ii3 = [10+81*2:9+81*3];  % Infant-to-Infant Alpha
aa1 = [10+81*3:9+81*4];  % Adult-to-Adult Delta
aa2 = [10+81*4:9+81*5];  % Adult-to-Adult Theta
aa3 = [10+81*5:9+81*6];  % Adult-to-Adult Alpha
ai1 = [10+81*6:9+81*7];  % Adult-to-Infant Delta
ai2 = [10+81*7:9+81*8];  % Adult-to-Infant Theta
ai3 = [10+81*8:9+81*9];  % Adult-to-Infant Alpha
ia1 = [10+81*9:9+81*10];  % Infant-to-Adult Delta
ia2 = [10+81*10:9+81*11]; % Infant-to-Adult Theta
ia3 = [10+81*11:9+81*12]; % Infant-to-Adult Alpha

% Identify self-connections (diagonals) to exclude
block_size = 81;
num_blocks = 12;
initial_cols = 10;
diag_indices = [1, 11, 21, 31, 41, 51, 61, 71, 81];
column_indices = [];

for j = 1:num_blocks
    for d = diag_indices
        col_index = initial_cols + (j-1) * block_size + d;
        column_indices = [column_indices, col_index];
    end
end

%% Analyze specific connectivity matrix
% Choose which connectivity matrix to analyze (edit as needed)
% For example, to analyze Adult-to-Infant Alpha connectivity:
listi = ai3;

% Apply square root transformation to normalize data
data = sqrt(data(:, listi));
titles = titles(listi);

%% Compare real data with surrogate distribution by condition

% Initialize arrays for storing results
significant_titles = {};
group1_idx = find(data(:, 6) == 1);  % Full gaze
group2_idx = find(data(:, 6) == 2);  % Partial gaze
group3_idx = find(data(:, 6) == 3);  % No gaze
group4_idx = 1:size(data, 1);       % All data

% Initialize p-value arrays
p_group1 = zeros(size(data, 2), 1);
p_group2 = zeros(size(data, 2), 1);
p_group3 = zeros(size(data, 2), 1);
p_group4 = zeros(size(data, 2), 1);

% Calculate mean connectivity for each group
mean_data_group1 = zeros(1, size(data, 2));
mean_data_group2 = zeros(1, size(data, 2));
mean_data_group3 = zeros(1, size(data, 2));
mean_data_group4 = zeros(1, size(data, 2));

for j = 1:size(data, 2)
    mean_data_group1(j) = nanmean(data(group1_idx, j));
    mean_data_group2(j) = nanmean(data(group2_idx, j));
    mean_data_group3(j) = nanmean(data(group3_idx, j));
    mean_data_group4(j) = nanmean(data(group4_idx, j));
end

% Initialize arrays for surrogate means
mean_surr_group1 = zeros(length(data_surr), size(data, 2));
mean_surr_group2 = zeros(length(data_surr), size(data, 2));
mean_surr_group3 = zeros(length(data_surr), size(data, 2));
mean_surr_group4 = zeros(length(data_surr), size(data, 2));

% Calculate mean connectivity for each surrogate dataset
fprintf('Processing surrogate datasets...\n');
for count = 1:length(data_surr)
    if mod(count, 10) == 0
        fprintf('Processed %d of %d surrogate datasets\n', count, length(data_surr));
    end
    
    % Get current surrogate dataset
    surr = data_surr{count};
    surr2 = sqrt(surr(:, listi));
    surr2(cutlist, :) = [];
    
    % Calculate mean for each group
    mean_surr_group1(count, :) = nanmean(surr2(group1_idx, :), 1);
    mean_surr_group2(count, :) = nanmean(surr2(group2_idx, :), 1);
    mean_surr_group3(count, :) = nanmean(surr2(group3_idx, :), 1);
    mean_surr_group4(count, :) = nanmean(surr2(group4_idx, :), 1);
end

% Calculate p-values (proportion of surrogate means exceeding real means)
for j = 1:size(data, 2)
    p_group1(j) = (sum(mean_surr_group1(:, j) > mean_data_group1(j)) + 1) / (size(mean_surr_group1, 1) + 1);
    p_group2(j) = (sum(mean_surr_group2(:, j) > mean_data_group2(j)) + 1) / (size(mean_surr_group1, 1) + 1);
    p_group3(j) = (sum(mean_surr_group3(:, j) > mean_data_group3(j)) + 1) / (size(mean_surr_group1, 1) + 1);
    p_group4(j) = (sum(mean_surr_group4(:, j) > mean_data_group4(j)) + 1) / (size(mean_surr_group4, 1) + 1);
end

% Apply FDR correction
p_group1c = mafdr(p_group1, 'BHFDR', true);
p_group2c = mafdr(p_group2, 'BHFDR', true);
p_group3c = mafdr(p_group3, 'BHFDR', true);
p_group4c = mafdr(p_group4, 'BHFDR', true);

% Find significant connections
s1 = find(p_group1c < 0.05);  % Significant in Full gaze condition
s2 = find(p_group2c < 0.05);  % Significant in Partial gaze condition
s3 = find(p_group3c < 0.05);  % Significant in No gaze condition
s4 = find(p_group4c < 0.05);  % Significant across all data

% Combine significant connections
stronglist = union(s1, s2);
stronglist = union(stronglist, s3);

% Display summary of significant connections
fprintf('\nSignificant connections after FDR correction:\n');
fprintf('Full gaze condition: %d connections\n', length(s1));
fprintf('Partial gaze condition: %d connections\n', length(s2));
fprintf('No gaze condition: %d connections\n', length(s3));
fprintf('All data combined: %d connections\n', length(s4));

% Save significant connections
save_path = fullfile(base_path, ['stronglistfdr5_', lower(connectivity_type), '_', connectionTypes{3}, frequencyBands{3}, '.mat']);
save(save_path, 'stronglist', 's1', 's2', 's3', 's4');

%% Visualize connectivity strength vs. surrogate distribution

% Set up figure parameters
titles_plot = {'II GPDC', 'AA GPDC', 'AI GPDC', 'IA GPDC'};
colorlist = {[252/255, 140/255, 90/255], ...
            [226/255, 90/255, 80/255], ...
            [75/255, 116/255, 178/255], ...
            [144/255, 190/255, 224/255]};

% Define which matrices to plot
list_indices = {ii3, aa3, ai3, ia3};  % Alpha band for all four quadrants

for idx = 1:4
    % Select current connectivity matrix
    listi = list_indices{idx};
    
    % Remove diagonal elements for within-brain connectivity
    if idx <= 2
        listi([1:10:81]) = [];
    end
    
    % Initialize storage for surrogate means
    mean_data = zeros(length(data_surr), length(listi));
    
    % Calculate means for each surrogate dataset
    fprintf('Processing surrogate data for visualization %d/4...\n', idx);
    for i = 1:length(data_surr)
        if mod(i, 20) == 0
            fprintf('  Processed %d of %d surrogate datasets\n', i, length(data_surr));
        end
        
        surr_data = sqrt(data_surr{i});
        surr_data(cutlist, :) = [];
        mean_data(i, :) = mean(surr_data(:, listi), 1);
    end
    
    % Calculate mean and percentiles of surrogate distribution
    mean_all = mean(mean_data, 1);
    prct_low = prctile(mean_data, 2.5, 1);
    prct_high = prctile(mean_data, 97.5, 1);
    
    % Calculate real data mean
    original_data = sqrt(data);
    original_data_mean = mean(original_data(:, listi), 1);
    
    % Sort by strength
    [~, b] = sort(original_data_mean);
    
    % Count connections exceeding 97.5% surrogate threshold
    exceed_count = sum(original_data_mean > prct_high);
    total_count = length(original_data_mean);
    exceed_percentage = (exceed_count / total_count) * 100;
    
    % Create figure
    figure('Position', [100, 100, 1200, 600]);
    hold on;
    
    % Plot 95% confidence interval for surrogate data
    fill([1:length(prct_low(b)), fliplr(1:length(prct_high(b)))], ...
         [prct_low(b), fliplr(prct_high(b))], ...
         [200/255, 200/255, 200/255], 'EdgeColor', 'none');
    
    % Plot mean of surrogate data
    plot(mean_all(b), 'LineWidth', 3, 'Color', [0/255, 0/255, 0/255]);
    
    % Plot mean of real data
    plot(original_data_mean(b), 'LineWidth', 3, 'Color', colorlist{idx});
    
    % Formatting
    xlabel('Connections (Strength-Ranked)', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Strength', 'FontSize', 16, 'FontWeight', 'bold');
    title(titles_plot{idx}, 'FontSize', 20, 'FontWeight', 'bold');
    
    % Add annotation about significant connections
    text(10, max(original_data_mean)*0.9, ...
         sprintf('%.1f%% (%d/%d) significant connections', ...
                exceed_percentage, exceed_count, total_count), ...
         'FontSize', 14, 'FontWeight', 'bold');
    
    % Format axes
    xlim([0, length(listi)]);
    ax = gca;
    ax.LineWidth = 2;
    ax.FontSize = 14;
    ax.FontWeight = 'bold';
    
    hold off;
    
    % Save figure
    saveas(gcf, fullfile(base_path, 'figures', [lower(connectivity_type), '_', titles_plot{idx}, '.png']));
end

%% Test condition effects on significant connections using linear mixed-effects models

% Load significant connections list (if not already loaded)
if ~exist('stronglist', 'var')
    load(fullfile(base_path, ['stronglistfdr5_', lower(connectivity_type), '_AI.mat']));
    stronglist = s4;  % Use connections significant across all data
end

% Initialize arrays for t-values and p-values
tValueCondition = zeros(length(stronglist), 1);
pValueCondition = ones(length(stronglist), 1);

% Test each significant connection
fprintf('\nTesting condition effects on significant connections...\n');
for i = 1:length(stronglist)
    % Extract current connection
    Y = data(:, stronglist(i));
    
    % Create table for linear mixed-effects model
    tbl = table(ID, zscore(learning), zscore(atten), zscore(AGE), SEX, COUNTRY, ...
               zscore(Y), blocks, CONDGROUP, 'VariableNames', ...
               {'ID', 'learning', 'atten', 'AGE', 'SEX', 'COUNTRY', 'Y', 'block', 'CONDGROUP'});
    
    % Fit linear mixed-effects model with condition as predictor
    lme = fitlme(tbl, 'Y ~ AGE + SEX + CONDGROUP + COUNTRY + (1|ID)');
    
    % Extract coefficient for condition effect (CONDGROUP_2 = Partial vs. Full gaze)
    coeffIdx = strcmp(lme.Coefficients.Name, 'CONDGROUP_2');
    tValueCondition(i) = lme.Coefficients.tStat(coeffIdx);
    pValueCondition(i) = lme.Coefficients.pValue(coeffIdx);
end

% Apply FDR correction
qValueCondition = mafdr(pValueCondition, 'BHFDR', true);

% Display significant results
sig_idx = find(qValueCondition < 0.05);
fprintf('\nSignificant condition effects after FDR correction:\n');
for i = 1:length(sig_idx)
    fprintf('Connection %s: t = %.2f, p = %.4f, q = %.4f\n', ...
           titles{stronglist(sig_idx(i))}, ...
           tValueCondition(sig_idx(i)), ...
           pValueCondition(sig_idx(i)), ...
           qValueCondition(sig_idx(i)));
end

% Calculate Bonferroni threshold for comparison
df = size(data, 1) - 2;  % Degrees of freedom
p_bonf = 0.05 / length(stronglist);  % Bonferroni corrected p-value
t_threshold = tinv(1 - p_bonf/2, df);

fprintf('\nBonferroni threshold: t(%.0f) = %.4f (p = %.6f)\n', df, t_threshold, p_bonf);
fprintf('Top 5 t-values: ');
[sorted_t, idx] = sort(abs(tValueCondition), 'descend');
for i = 1:min(5, length(sorted_t))
    fprintf('%.4f ', sorted_t(i));
end
fprintf('\n');

%% Examine specific connection of interest (example)
% Analyze the 12th connection as an example (could be any specific connection of interest)
if length(stronglist) >= 12
    connection_idx = 12;
    
    % Extract data for this connection
    Y = data(:, connection_idx);
    
    % Create table for linear mixed-effects model
    tbl = table(ID, zscore(learning), zscore(atten), zscore(AGE), SEX, COUNTRY, ...
               zscore(Y), blocks, CONDGROUP, 'VariableNames', ...
               {'ID', 'learning', 'atten', 'AGE', 'SEX', 'COUNTRY', 'Y', 'block', 'CONDGROUP'});
    
    % Fit linear mixed-effects model
    lme = fitlme(tbl, 'Y ~ AGE + SEX + CONDGROUP + COUNTRY + (1|ID)');
    
    % Display results
    fprintf('\nAnalysis of specific connection of interest (%s):\n', titles{connection_idx});
    disp(lme.Coefficients);
    
    % Visualize this connection by condition
    figure('Position', [100, 100, 800, 600]);
    boxplot(Y, CONDGROUP, 'Labels', {'Full Gaze', 'Partial Gaze', 'No Gaze'});
    hold on;
    plot(ones(length(g1), 1) + 0.1*randn(length(g1), 1), Y(g1), 'r.', 'MarkerSize', 15);
    plot(2*ones(length(g2), 1) + 0.1*randn(length(g2), 1), Y(g2), 'b.', 'MarkerSize', 15);
    plot(3*ones(length(g3), 1) + 0.1*randn(length(g3), 1), Y(g3), 'g.', 'MarkerSize', 15);
    hold off;
    ylabel('Connectivity Strength', 'FontSize', 14, 'FontWeight', 'bold');
    title(['Effect of Gaze Condition on ', titles{connection_idx}], 'FontSize', 16, 'FontWeight', 'bold');
    ax = gca;
    ax.LineWidth = 2;
    ax.FontSize = 14;
    ax.FontWeight = 'bold';
    
    % Save figure
    saveas(gcf, fullfile(base_path, 'figures', [lower(connectivity_type), '_connection_', num2str(connection_idx), '.png']));
end

fprintf('\nAnalysis complete.\n');
