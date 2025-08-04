%% Neural Entrainment Statistical Analysis
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Statistical analysis of entrainment between speech envelopes and neural oscillations
% in infant EEG data across different experimental conditions (gaze manipulation)
%
% This script processes pre-computed entrainment measures to:
% 1. Analyze statistical significance using permutation testing against surrogate data
% 2. Apply FDR correction for multiple comparisons across channels and frequency bands
% 3. Visualize entrainment effects and create statistical heatmaps
% 4. Identify significant neural-speech entrainment patterns

%% Initialize environment
clc;
clear all;

% Set paths
base_dir = '/path/to/data/';
analysis_dir = fullfile(base_dir, 'analysis');
figures_dir = fullfile(base_dir, 'figures');

% Create figures directory if it doesn't exist
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% Load and prepare entrainment data

% Load entrainment data from first phrase only (as used in main analysis)
phrase = 1;
[num_data, text_data] = xlsread(fullfile(analysis_dir, 'ENTRAINTABLE.xlsx'));

% Select only data from specified phrase
num_data = num_data(num_data(:,3) == phrase,:);

% Separate data by experimental condition
cond1_indices = find(num_data(:,2) == 1);  % Full gaze condition
cond2_indices = find(num_data(:,2) == 2);  % Partial gaze condition
cond3_indices = find(num_data(:,2) == 3);  % No gaze condition

fprintf('Loaded entrainment data: %d participants\n', size(num_data, 1));
fprintf('Full gaze: %d observations, Partial gaze: %d observations, No gaze: %d observations\n', ...
        length(cond1_indices), length(cond2_indices), length(cond3_indices));

%% Load surrogate data from permutation testing

load(fullfile(base_dir, 'ENTRIANSURR.mat'), 'permutable');

% Initialize matrices for surrogate data means by condition
surrogate_mean1 = [];
surrogate_mean2 = [];
surrogate_mean3 = [];

fprintf('Processing %d surrogate datasets...\n', length(permutable));

% Process surrogate data for each permutation
for i = 1:length(permutable)
    if ~isempty(permutable{i})
        % Extract numerical data from table
        tmp_data = permutable{i};
        tmp_data = table2array(tmp_data(:,2:end));
        
        % Take absolute value of small correlations for consistency
        tmp_data(abs(tmp_data) < 1) = abs(tmp_data(abs(tmp_data) < 1));
        
        % Select data from specified phrase
        tmp_data = tmp_data(tmp_data(:,3) == phrase,:);
        
        % Group by condition
        tmp_cond1 = tmp_data(tmp_data(:,2) == 1,:);
        tmp_cond2 = tmp_data(tmp_data(:,2) == 2,:);
        tmp_cond3 = tmp_data(tmp_data(:,2) == 3,:);
        
        % Calculate mean for each condition and add to collection
        if ~isempty(tmp_cond1)
            surrogate_mean1 = [surrogate_mean1; nanmean(tmp_cond1)];
        end
        if ~isempty(tmp_cond2)
            surrogate_mean2 = [surrogate_mean2; nanmean(tmp_cond2)];
        end
        if ~isempty(tmp_cond3)
            surrogate_mean3 = [surrogate_mean3; nanmean(tmp_cond3)];
        end
    end
end

fprintf('Processed surrogate data: %d iterations per condition\n', size(surrogate_mean1, 1));

%% Define column indices for peak and lag measures

% Column indices for entrainment measures by region and frequency band
% Based on the structure: [block, condition, phrase, alpha_peak(9), alpha_lag(9), 
%                          theta_peak(9), theta_lag(9), delta_peak(9), delta_lag(9)]
peak_columns = [4:12, 22:30, 40:48];  % Alpha, Theta, Delta peak correlation values
lag_columns = [13:21, 31:39, 49:57];  % Alpha, Theta, Delta time lag values

% Define frequency band and channel names for reporting
frequency_bands = {'Alpha', 'Theta', 'Delta'};
channels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};

%% Process real entrainment data

% Take absolute value of small correlations for consistency with surrogate data
num_data(abs(num_data) < 1) = abs(num_data(abs(num_data) < 1));

% Calculate means for each condition
real_mean1 = nanmean(num_data(num_data(:,2) == 1,:));  % Full gaze
real_mean2 = nanmean(num_data(num_data(:,2) == 2,:));  % Partial gaze
real_mean3 = nanmean(num_data(num_data(:,2) == 3,:));  % No gaze

%% Statistical testing for Condition 1 (Full gaze) - peak values

sig_peak_cols1 = [];
peak_pvalues1 = [];

fprintf('\nTesting peak correlation significance for Full gaze condition...\n');

for i = 1:length(peak_columns)
    col = peak_columns(i);
    % Calculate p-value (proportion of surrogate values >= real value)
    p_value = (sum(surrogate_mean1(:,col) >= real_mean1(col)) + 1) / (size(surrogate_mean1, 1) + 1);
    
    sig_peak_cols1 = [sig_peak_cols1, col];
    peak_pvalues1 = [peak_pvalues1, p_value];
end

%% Statistical testing for Condition 1 (Full gaze) - lag values

sig_lag_cols1 = [];
lag_outliers1 = {};
lag_pvalues1 = [];

fprintf('Testing lag value significance for Full gaze condition...\n');

for i = 1:length(lag_columns)
    col = lag_columns(i);
    
    % Calculate percentile thresholds for two-tailed test
    lower_percentile = prctile(surrogate_mean1(:,col), 2.5);
    upper_percentile = prctile(surrogate_mean1(:,col), 97.5);
    
    if real_mean1(col) < lower_percentile
        p_value = (sum(surrogate_mean1(:,col) <= real_mean1(col)) + 1) / (size(surrogate_mean1, 1) + 1);
        outlier_status = "< 2.5%";
    elseif real_mean1(col) > upper_percentile
        p_value = (sum(surrogate_mean1(:,col) >= real_mean1(col)) + 1) / (size(surrogate_mean1, 1) + 1);
        outlier_status = "> 97.5%";
    else
        p_value = 2 * min(sum(surrogate_mean1(:,col) <= real_mean1(col)), ...
            sum(surrogate_mean1(:,col) >= real_mean1(col))) / size(surrogate_mean1, 1);
        outlier_status = "Not significant";
    end
    
    sig_lag_cols1 = [sig_lag_cols1, col];
    lag_outliers1 = [lag_outliers1, outlier_status];
    lag_pvalues1 = [lag_pvalues1, p_value];
end

% Apply FDR correction for multiple comparisons
q_peak1 = mafdr(peak_pvalues1, 'BHFDR', true);
q_lag1 = mafdr(lag_pvalues1, 'BHFDR', true);

%% Statistical testing for Condition 2 (Partial gaze) - peak values

sig_peak_cols2 = [];
peak_pvalues2 = [];

fprintf('Testing peak correlation significance for Partial gaze condition...\n');

for i = 1:length(peak_columns)
    col = peak_columns(i);
    p_value = (sum(surrogate_mean2(:,col) >= real_mean2(col)) + 1) / (size(surrogate_mean2, 1) + 1);
    
    sig_peak_cols2 = [sig_peak_cols2, col];
    peak_pvalues2 = [peak_pvalues2, p_value];
end

%% Statistical testing for Condition 2 (Partial gaze) - lag values

sig_lag_cols2 = [];
lag_outliers2 = {};
lag_pvalues2 = [];

for i = 1:length(lag_columns)
    col = lag_columns(i);
    lower_percentile = prctile(surrogate_mean2(:,col), 2.5);
    upper_percentile = prctile(surrogate_mean2(:,col), 97.5);
    
    if real_mean2(col) < lower_percentile
        p_value = (sum(surrogate_mean2(:,col) <= real_mean2(col)) + 1) / (size(surrogate_mean2, 1) + 1);
        outlier_status = "< 2.5%";
    elseif real_mean2(col) > upper_percentile
        p_value = (sum(surrogate_mean2(:,col) >= real_mean2(col)) + 1) / (size(surrogate_mean2, 1) + 1);
        outlier_status = "> 97.5%";
    else
        p_value = 2 * min(sum(surrogate_mean2(:,col) <= real_mean2(col)), ...
            sum(surrogate_mean2(:,col) >= real_mean2(col))) / size(surrogate_mean2, 1);
        outlier_status = "Not significant";
    end
    
    sig_lag_cols2 = [sig_lag_cols2, col];
    lag_outliers2 = [lag_outliers2, outlier_status];
    lag_pvalues2 = [lag_pvalues2, p_value];
end

% Apply FDR correction
q_peak2 = mafdr(peak_pvalues2, 'BHFDR', true);
q_lag2 = mafdr(lag_pvalues2, 'BHFDR', true);

%% Statistical testing for Condition 3 (No gaze) - peak values

sig_peak_cols3 = [];
peak_pvalues3 = [];

fprintf('Testing peak correlation significance for No gaze condition...\n');

for i = 1:length(peak_columns)
    col = peak_columns(i);
    p_value = (sum(surrogate_mean3(:,col) >= real_mean3(col)) + 1) / (size(surrogate_mean3, 1) + 1);
    
    sig_peak_cols3 = [sig_peak_cols3, col];
    peak_pvalues3 = [peak_pvalues3, p_value];
end

%% Statistical testing for Condition 3 (No gaze) - lag values

sig_lag_cols3 = [];
lag_outliers3 = {};
lag_pvalues3 = [];

for i = 1:length(lag_columns)
    col = lag_columns(i);
    lower_percentile = prctile(surrogate_mean3(:,col), 2.5);
    upper_percentile = prctile(surrogate_mean3(:,col), 97.5);
    
    if real_mean3(col) < lower_percentile
        p_value = (sum(surrogate_mean3(:,col) <= real_mean3(col)) + 1) / (size(surrogate_mean3, 1) + 1);
        outlier_status = "< 2.5%";
    elseif real_mean3(col) > upper_percentile
        p_value = (sum(surrogate_mean3(:,col) >= real_mean3(col)) + 1) / (size(surrogate_mean3, 1) + 1);
        outlier_status = "> 97.5%";
    else
        p_value = 2 * min(sum(surrogate_mean3(:,col) <= real_mean3(col)), ...
            sum(surrogate_mean3(:,col) >= real_mean3(col))) / size(surrogate_mean3, 1);
        outlier_status = "Not significant";
    end
    
    sig_lag_cols3 = [sig_lag_cols3, col];
    lag_outliers3 = [lag_outliers3, outlier_status];
    lag_pvalues3 = [lag_pvalues3, p_value];
end

% Apply FDR correction
q_peak3 = mafdr(peak_pvalues3, 'BHFDR', true);
q_lag3 = mafdr(lag_pvalues3, 'BHFDR', true);

%% Display significant results

% Display results for Condition 1 (Full gaze)
fprintf('\n=== RESULTS FOR FULL GAZE CONDITION ===\n');
fprintf('Significant peak correlations after FDR correction (q <= 0.05):\n');
sig_count = 0;
for i = 1:length(sig_peak_cols1)
    col = sig_peak_cols1(i);
    if q_peak1(i) <= 0.05
        fprintf('%s (Column %d): q = %.4f\n', text_data{1, col + 1}, col, q_peak1(i));
        sig_count = sig_count + 1;
    end
end
if sig_count == 0
    fprintf('No significant peak correlations found.\n');
end

fprintf('\nSignificant lag values after FDR correction (q <= 0.05):\n');
sig_count = 0;
for i = 1:length(sig_lag_cols1)
    col = sig_lag_cols1(i);
    if q_lag1(i) <= 0.05
        fprintf('%s (Column %d): %s, q = %.4f\n', text_data{1, col + 1}, col, lag_outliers1{i}, q_lag1(i));
        sig_count = sig_count + 1;
    end
end
if sig_count == 0
    fprintf('No significant lag values found.\n');
end

% Display results for Condition 2 (Partial gaze)
fprintf('\n=== RESULTS FOR PARTIAL GAZE CONDITION ===\n');
fprintf('Significant peak correlations after FDR correction (q <= 0.05):\n');
sig_count = 0;
for i = 1:length(sig_peak_cols2)
    col = sig_peak_cols2(i);
    if q_peak2(i) <= 0.05
        fprintf('%s (Column %d): q = %.4f\n', text_data{1, col + 1}, col, q_peak2(i));
        sig_count = sig_count + 1;
    end
end
if sig_count == 0
    fprintf('No significant peak correlations found.\n');
end

fprintf('\nSignificant lag values after FDR correction (q <= 0.05):\n');
sig_count = 0;
for i = 1:length(sig_lag_cols2)
    col = sig_lag_cols2(i);
    if q_lag2(i) <= 0.05
        fprintf('%s (Column %d): %s, q = %.4f\n', text_data{1, col + 1}, col, lag_outliers2{i}, q_lag2(i));
        sig_count = sig_count + 1;
    end
end
if sig_count == 0
    fprintf('No significant lag values found.\n');
end

% Display results for Condition 3 (No gaze)
fprintf('\n=== RESULTS FOR NO GAZE CONDITION ===\n');
fprintf('Significant peak correlations after FDR correction (q <= 0.05):\n');
sig_count = 0;
for i = 1:length(sig_peak_cols3)
    col = sig_peak_cols3(i);
    if q_peak3(i) <= 0.05
        fprintf('%s (Column %d): q = %.4f\n', text_data{1, col + 1}, col, q_peak3(i));
        sig_count = sig_count + 1;
    end
end
if sig_count == 0
    fprintf('No significant peak correlations found.\n');
end

fprintf('\nSignificant lag values after FDR correction (q <= 0.05):\n');
sig_count = 0;
for i = 1:length(sig_lag_cols3)
    col = sig_lag_cols3(i);
    if q_lag3(i) <= 0.05
        fprintf('%s (Column %d): %s, q = %.4f\n', text_data{1, col + 1}, col, lag_outliers3{i}, q_lag3(i));
        sig_count = sig_count + 1;
    end
end
if sig_count == 0
    fprintf('No significant lag values found.\n');
end

%% Create visualization for statistical results

% Create heatmap visualization of effect sizes across channels and frequency bands
figure('Position', [100, 100, 1600, 500]);
create_statistical_matrices(peak_pvalues1, q_peak1, real_mean1, surrogate_mean1, peak_columns, ...
                          peak_pvalues2, q_peak2, real_mean2, surrogate_mean2, ...
                          peak_pvalues3, q_peak3, real_mean3, surrogate_mean3);

% Save figure
saveas(gcf, fullfile(figures_dir, 'entrainment_statistical_heatmap.png'));
saveas(gcf, fullfile(figures_dir, 'entrainment_statistical_heatmap.fig'));

%% Save statistical results

% Create summary table of results
results_summary = struct();
results_summary.conditions = {'Full_Gaze', 'Partial_Gaze', 'No_Gaze'};
results_summary.peak_pvalues = {peak_pvalues1, peak_pvalues2, peak_pvalues3};
results_summary.peak_qvalues = {q_peak1, q_peak2, q_peak3};
results_summary.lag_pvalues = {lag_pvalues1, lag_pvalues2, lag_pvalues3};
results_summary.lag_qvalues = {q_lag1, q_lag2, q_lag3};
results_summary.peak_columns = peak_columns;
results_summary.lag_columns = lag_columns;
results_summary.real_means = {real_mean1, real_mean2, real_mean3};
results_summary.surrogate_means = {surrogate_mean1, surrogate_mean2, surrogate_mean3};

save(fullfile(base_dir, 'entrainment_statistical_results.mat'), 'results_summary');

fprintf('\nAnalysis complete. Results saved to entrainment_statistical_results.mat\n');

%% Helper Functions

function cohens_d = calculate_cohens_d(real_mean, surrogate_mean)
    % Function to calculate Cohen's d as effect size metric
    [n_samples, n_cols] = size(surrogate_mean);
    cohens_d = zeros(1, n_cols);
    
    for i = 1:n_cols
        group1 = repmat(real_mean(i), n_samples, 1);
        group2 = surrogate_mean(:,i);
        
        % Use standard deviation of surrogate data as baseline
        s2 = std(group2);
        
        % Calculate effect size (Cohen's d)
        if s2 > 0
            d = (mean(group1) - mean(group2)) / s2;
        else
            d = 0;
        end
        
        cohens_d(i) = d;
    end
end

function h = plot_matrix(subplot_idx, p_values, q_values, real_mean, surrogate_mean, cols)
    % Function to plot a single matrix of results
    h = subplot(1, 3, subplot_idx);
    
    % Calculate Cohen's d effect sizes
    cohens_d = calculate_cohens_d(real_mean(cols), surrogate_mean(:,cols));
    
    % Reshape data for 9x3 matrix (channels x frequency bands)
    cohens_d_matrix = reshape(cohens_d, [9, 3]);
    q_values_matrix = reshape(q_values, [9, 3]);
    
    % Reverse column order for display (Delta, Theta, Alpha -> Alpha, Theta, Delta)
    cohens_d_matrix = fliplr(cohens_d_matrix);
    q_values_matrix = fliplr(q_values_matrix);
    
    % Display the heatmap
    imagesc(cohens_d_matrix);
    
    % Add statistical significance markers
    [rows, cols] = size(cohens_d_matrix);
    hold on;
    for i = 1:rows
        for j = 1:cols
            % Add significance stars based on q-value
            if q_values_matrix(i,j) < 0.05
                text(j, i, '*', 'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle', 'Color', 'k', ...
                     'FontSize', 50, 'FontWeight', 'bold');
            end
        end
    end
    hold off;
    
    % Customize plot appearance
    set(gca, 'XTick', 1:cols, 'YTick', 1:rows);
    set(gca, 'XTickLabel', {'Alpha', 'Theta', 'Delta'});
    set(gca, 'YTickLabel', {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'});
    
    % Add condition labels
    condition_labels = {'Full Gaze', 'Partial Gaze', 'No Gaze'};
    title(condition_labels{subplot_idx}, 'FontSize', 18, 'FontWeight', 'bold');
    
    if subplot_idx == 1
        ylabel('Channel', 'FontSize', 16, 'FontWeight', 'bold');
    end
    if subplot_idx == 2
        xlabel('Frequency Band', 'FontSize', 20, 'FontWeight', 'bold');
    end
    
    % Enhance visualization
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'TickDir', 'out');
    set(gca, 'LineWidth', 2);
end

function create_statistical_matrices(peak_p1, q_peak1, real_mean1, surrogate_mean1, peak_columns, ...
                                  peak_p2, q_peak2, real_mean2, surrogate_mean2, ...
                                  peak_p3, q_peak3, real_mean3, surrogate_mean3)
    % Function to create statistical matrices for all conditions
    
    % Plot statistical matrices for each condition
    h1 = plot_matrix(1, peak_p1, q_peak1, real_mean1, surrogate_mean1, peak_columns);
    h2 = plot_matrix(2, peak_p2, q_peak2, real_mean2, surrogate_mean2, peak_columns);
    h3 = plot_matrix(3, peak_p3, q_peak3, real_mean3, surrogate_mean3, peak_columns);
    
    % Create colormap and colorbar
    colormap(redblue(256));
    c = colorbar;
    
    % Set unified color scale across subplots
    max_abs_d = max([max(abs(get(h1, 'CLim'))), max(abs(get(h2, 'CLim'))), max(abs(get(h3, 'CLim')))]);
    set([h1 h2 h3], 'CLim', [-max_abs_d, max_abs_d]);
    
    % Customize colorbar
    c.LineWidth = 2;
    c.FontSize = 14;
    c.FontWeight = 'bold';
    c.Label.String = 'Cohen''s d (Effect Size)';
    c.Label.FontSize = 16;
    c.Label.FontWeight = 'bold';
    
    % Position the colorbar
    c_pos = c.Position;
    c.Position = [c_pos(1)+0.05 c_pos(2) c_pos(3) c_pos(4)];
    
    % Add main title
    sgtitle('Neural-Speech Entrainment: Statistical Significance Testing', ...
            'FontSize', 20, 'FontWeight', 'bold');
end

function cmap = redblue(m)
    % Custom colormap function for visualization (red-white-blue)
    if nargin < 1
        m = 64; 
    end
    
    % Create gradient from blue through white to red
    if mod(m,2) == 0
        % Even number of colors
        n = m/2;
        r = [linspace(0, 1, n), ones(1, n)];
        g = [linspace(0, 1, n), linspace(1, 0, n)];
        b = [ones(1, n), linspace(1, 0, n)];
    else
        % Odd number of colors
        n = (m-1)/2;
        r = [linspace(0, 1, n), 1, ones(1, n)];
        g = [linspace(0, 1, n), 1, linspace(1, 0, n)];
        b = [ones(1, n), 1, linspace(1, 0, n)];
    end
    
    cmap = [r', g', b'];
end
