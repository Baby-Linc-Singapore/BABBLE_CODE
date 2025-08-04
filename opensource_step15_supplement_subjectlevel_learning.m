%% Subject-Level Analysis of Learning Patterns Across Gaze Conditions
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Analyze how infants' learning performance varies across different gaze conditions
%          and identify distinct learning patterns at the subject level
% This script performs:
% 1. Subject-level aggregation of learning data across conditions
% 2. Classification of subjects into learning pattern groups
% 3. Visualization of learning performance in 3D space
% 4. Statistical analysis of group differences

%% Load and prepare data
clear all

% Load behavioral data
[data, headers] = xlsread('behaviour.xlsx');

% Extract demographic and performance variables
Country = data(:,1);   % Data collection site (1=UK, 2=Singapore)
ID = data(:,2);        % Subject ID
AGE = data(:,3);       % Age in months
SEX = data(:,4);       % Sex (1=female, 2=male)
learning = data(:,7);  % Learning performance measure (seconds)
Attention = data(:,9); % Attention measure
blocks = data(:,5);    % Experimental block (1-3)
cond = data(:,6);      % Experimental condition:
                       % 1 = Full gaze
                       % 2 = Partial gaze
                       % 3 = No gaze

% Find subjects with data from each condition
unique_ids = unique(ID);
id_col = [];
mean_cond1 = [];  % Full gaze
mean_cond2 = [];  % Partial gaze 
mean_cond3 = [];  % No gaze

% For each unique subject
for i = 1:length(unique_ids)
    current_id = unique_ids(i);
    id_indices = find(ID == current_id);
    
    % Get all values for each condition (across all blocks)
    cond1_vals = learning(id_indices(cond(id_indices) == 1));
    cond2_vals = learning(id_indices(cond(id_indices) == 2));
    cond3_vals = learning(id_indices(cond(id_indices) == 3));
    
    % Only include subject if they have data for all conditions
    if ~isempty(cond1_vals) && ~isempty(cond2_vals) && ~isempty(cond3_vals)
        id_col = [id_col; current_id];
        mean_cond1 = [mean_cond1; mean(cond1_vals, 'omitnan')];
        mean_cond2 = [mean_cond2; mean(cond2_vals, 'omitnan')];
        mean_cond3 = [mean_cond3; mean(cond3_vals, 'omitnan')];
    end
end

% Create subject-level results matrix
results_matrix = [id_col mean_cond1 mean_cond2 mean_cond3];

% Display summary
fprintf('Subject-level data matrix created with %d subjects.\n', size(results_matrix, 1));
fprintf('Matrix contains: ID, Mean_FullGaze, Mean_PartialGaze, Mean_NoGaze\n');

%% Classify subjects based on their learning patterns
% Extract condition-specific learning values
x = results_matrix(:,2);  % Full gaze
y = results_matrix(:,3);  % Partial gaze
z = results_matrix(:,4);  % No gaze
ids = results_matrix(:,1); % Subject IDs

% Initialize learning group classification
result = zeros(length(ids), 3);
for i = 1:length(ids)
    % Group 1: Full gaze advantage (full > partial > no)
    if x(i) > y(i) && x(i) > z(i) && x(i) > 0
        result(i, 1) = 1;
    % Group 2: Partial gaze advantage (partial > full > no)  
    elseif y(i) > x(i) && y(i) > z(i) && y(i) > 0
        result(i, 2) = 1;
    % Group 3: No gaze advantage (no > partial > full)
    elseif z(i) > x(i) && z(i) > y(i) && z(i) > 0
        result(i, 3) = 1;
    end
end

% Initialize variables for subject-level analysis
unique_ids = unique(ids);
n_subjects = length(unique_ids);
age_subj = zeros(n_subjects, 1);
sex_subj = zeros(n_subjects, 1);
country_subj = zeros(n_subjects, 1);
mean_learning_subj = zeros(n_subjects, 1);
group_cat = zeros(n_subjects, 1);  % Learning pattern group

% For each subject
for i = 1:n_subjects
    curr_id = unique_ids(i);
    id_indices = find(ID == curr_id);
    
    % Demographics
    age_subj(i) = AGE(id_indices(1));
    sex_subj(i) = SEX(id_indices(1));
    country_subj(i) = Country(id_indices(1));
    
    % Calculate mean learning across all conditions
    subj_trials = find(ids == curr_id);
    mean_learning_subj(i) = mean([mean_cond1(subj_trials), mean_cond2(subj_trials), mean_cond3(subj_trials)]);
    
    % Determine learning pattern group
    subj_result = result(i, :);
    
    if any(subj_result(:,1))
        group_cat(i) = 1;  % Full gaze advantage
    elseif any(subj_result(:,2))
        group_cat(i) = 2;  % Partial gaze advantage
    elseif any(subj_result(:,3))
        group_cat(i) = 3;  % No gaze advantage
    else
        group_cat(i) = 0;  % No clear pattern
    end
end

%% Calculate average learning performance per group
% Get subject IDs for each learning pattern group
full_adv_ids = id_col(group_cat == 1);   % Full gaze advantage
part_adv_ids = id_col(group_cat == 2);   % Partial gaze advantage
no_adv_ids = id_col(group_cat == 3);     % No gaze advantage

% Full gaze advantage group
full_adv_learning = [];
for i = 1:length(full_adv_ids)
    current_id = full_adv_ids(i);
    id_indices = find(ID == current_id);
    all_learning = learning(id_indices);
    full_adv_learning = [full_adv_learning; all_learning];
end
[mean_full_adv, std_full_adv] = deal(nanmean(full_adv_learning), nanstd(full_adv_learning));

% Partial gaze advantage group
part_adv_learning = [];
for i = 1:length(part_adv_ids)
    current_id = part_adv_ids(i);
    id_indices = find(ID == current_id);
    all_learning = learning(id_indices);
    part_adv_learning = [part_adv_learning; all_learning];
end
[mean_part_adv, std_part_adv] = deal(nanmean(part_adv_learning), nanstd(part_adv_learning));

% No gaze advantage group
no_adv_learning = [];
for i = 1:length(no_adv_ids)
    current_id = no_adv_ids(i);
    id_indices = find(ID == current_id);
    all_learning = learning(id_indices);
    no_adv_learning = [no_adv_learning; all_learning];
end
[mean_no_adv, std_no_adv] = deal(nanmean(no_adv_learning), nanstd(no_adv_learning));

% Display group statistics
fprintf('\nLearning pattern group statistics:\n');
fprintf('1. Full gaze advantage: %d subjects, Mean learning = %.2f (SD = %.2f)\n', ...
    length(full_adv_ids), mean_full_adv, std_full_adv);
fprintf('2. Partial gaze advantage: %d subjects, Mean learning = %.2f (SD = %.2f)\n', ...
    length(part_adv_ids), mean_part_adv, std_part_adv);
fprintf('3. No gaze advantage: %d subjects, Mean learning = %.2f (SD = %.2f)\n', ...
    length(no_adv_ids), mean_no_adv, std_no_adv);

%% Create categorical table for statistical modeling
% Create analysis table with proper categorical variables
T = table(mean_learning_subj(group_cat > 0), age_subj(group_cat > 0), ...
    categorical(sex_subj(group_cat > 0)), ...
    categorical(country_subj(group_cat > 0)), ...
    categorical(group_cat(group_cat > 0)), ...
    unique_ids(group_cat > 0), ...
    'VariableNames', {'mean_learning', 'age', 'sex', 'country', 'group', 'id'});

% Fit linear mixed effects model
lme = fitlme(T, 'mean_learning ~ age + sex + country + group');
fprintf('\nLinear mixed effects model results:\n');
disp(lme);

%% Create block-wise data matrix
% This aggregates learning performance for each condition by block
clear id_col block_col learning_cond1 learning_cond2 learning_cond3

% For each unique ID
unique_ids = unique(ID);
for i = 1:length(unique_ids)
    current_id = unique_ids(i);
    id_indices = find(ID == current_id);
    
    % For each block
    for block = 1:3
        block_indices = id_indices(blocks(id_indices) == block);
        
        % Get learning values for each condition
        learning_values = nan(1,3);
        has_all_values = true;
        
        for condition = 1:3
            cond_idx = block_indices(cond(block_indices) == condition);
            if ~isempty(cond_idx) && ~isnan(learning(cond_idx))
                learning_values(condition) = learning(cond_idx);
            else
                has_all_values = false;
                break;
            end
        end
        
        % Add row only if we have all three values
        if has_all_values
            id_col = [id_col; current_id];
            block_col = [block_col; block];
            learning_cond1 = [learning_cond1; learning_values(1)];
            learning_cond2 = [learning_cond2; learning_values(2)];
            learning_cond3 = [learning_cond3; learning_values(3)];
        end
    end
end

% Create the block-wise results matrix
results_matrix = [id_col block_col learning_cond1 learning_cond2 learning_cond3];

% Display summary
fprintf('\nBlock-wise data matrix created with %d rows.\n', size(results_matrix, 1));
fprintf('Matrix contains: ID, Block, Learning_FullGaze, Learning_PartialGaze, Learning_NoGaze\n');

%% Create 3D visualization of learning patterns
% Get learning values
x = results_matrix(:,3);  % Full gaze learning
y = results_matrix(:,4);  % Partial gaze learning
z = results_matrix(:,5);  % No gaze learning
ids = results_matrix(:,1); % Subject IDs

% Calculate mean learning per ID for coloring the plot
unique_ids = unique(ids);
id_means = zeros(size(ids));
for i = 1:length(unique_ids)
    id_idx = ids == unique_ids(i);
    curr_mean = mean([x(id_idx), y(id_idx), z(id_idx)], 'all');
    id_means(id_idx) = curr_mean;
end

% Create finer grid for smoother surface
[xi, yi] = meshgrid(linspace(min(x), max(x), 100), linspace(min(y), max(y), 100));

% Use natural interpolation for smoother surface
zi = griddata(x, y, z, xi, yi, 'v4');
ci = griddata(x, y, id_means, xi, yi, 'v4');

% Create smoother interpolation
F = scatteredInterpolant(x, y, id_means, 'natural');
ci = F(xi, yi);

% Apply smoothing using gaussian filter
smoothing_sigma = 10;
xi = imgaussfilt(xi, smoothing_sigma);
yi = imgaussfilt(yi, smoothing_sigma);
zi = imgaussfilt(zi, smoothing_sigma);
ci = imgaussfilt(ci, smoothing_sigma);

% Create the 3D surface plot
figure;
surf(xi, yi, zi, ci);
colorbar;
colormap('jet');

% Set font and axes properties
set(gca, 'FontName', 'Arial', 'FontSize', 20, 'FontWeight', 'bold', 'LineWidth', 2);

xlabel('Full-gaze learning / sec', 'FontName', 'Arial', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Partial-gaze learning / sec', 'FontName', 'Arial', 'FontSize', 20, 'FontWeight', 'bold');
zlabel('No-gaze learning / sec', 'FontName', 'Arial', 'FontSize', 20, 'FontWeight', 'bold');
title('3D Surface Plot of Learning Patterns', 'FontName', 'Arial', 'FontSize', 16, 'FontWeight', 'bold');

% Format colorbar
c = colorbar;
set(c, 'FontName', 'Arial', 'FontSize', 16, 'FontWeight', 'bold', 'LineWidth', 2);
ylabel(c, 'Overall learning / sec', 'FontName', 'Arial', 'FontSize', 16, 'FontWeight', 'bold');

grid on;


%% Statistical comparison of high vs. lower performers
% Initialize storage for p-values
percentiles = 50:10:90;
p_values = zeros(length(percentiles), 3);  % 3 columns for each condition

for i = 1:length(percentiles)
    percenta = percentiles(i);
    threshold = prctile(id_means, percenta);
    high_mean_idx = id_means >= threshold;
    
    % Count unique subjects above threshold
    high_ids = unique(id_col(high_mean_idx));
    num_unique_ids = length(high_ids);
    
    % Display sample sizes
    fprintf('\nPercentile %d%%: %d cases, %d unique subjects\n', ...
            percenta, sum(high_mean_idx), num_unique_ids);
    
    % Calculate and display means
    fprintf('Full Gaze: High = %.2f, Others = %.2f\n', ...
        mean(results_matrix(high_mean_idx,3)), mean(results_matrix(~high_mean_idx,3)));
    fprintf('Partial Gaze: High = %.2f, Others = %.2f\n', ...
        mean(results_matrix(high_mean_idx,4)), mean(results_matrix(~high_mean_idx,4)));
    fprintf('No Gaze: High = %.2f, Others = %.2f\n', ...
        mean(results_matrix(high_mean_idx,5)), mean(results_matrix(~high_mean_idx,5)));
    
    % Perform t-tests
    [~,p1] = ttest2(results_matrix(high_mean_idx,3), results_matrix(~high_mean_idx,3));
    [~,p2] = ttest2(results_matrix(high_mean_idx,4), results_matrix(~high_mean_idx,4));
    [~,p3] = ttest2(results_matrix(high_mean_idx,5), results_matrix(~high_mean_idx,5));
    
    % Store p-values
    p_values(i,:) = [p1, p2, p3];
    
    % Display p-values
    fprintf('P-values: Full=%.4e, Partial=%.4e, No=%.4e\n', p1, p2, p3);
end

% Apply FDR correction for multiple comparisons
all_p = p_values(:);
q_values = mafdr(all_p, 'BHFDR', true);
q_values = reshape(q_values, size(p_values));

% Display summary table
fprintf('\nFinal results summary:\n');
fprintf('Percentile\tCondition\tP-value\t\tQ-value\n');
for i = 1:length(percentiles)
    for j = 1:3
        fprintf('%d%%\t\tCond%d\t\t%.4e\t%.4e\n', ...
                percentiles(i), j, p_values(i,j), q_values(i,j));
    end
end

% Visualize p-values and q-values
figure;
subplot(1,2,1);
imagesc(p_values);
colormap('jet');
colorbar;
title('P-values');
xlabel('Conditions');
ylabel('Percentiles');
set(gca, 'XTick', 1:3, 'XTickLabel', {'Full', 'Partial', 'No'});
set(gca, 'YTick', 1:length(percentiles), 'YTickLabel', percentiles);

subplot(1,2,2);
imagesc(q_values);
colormap('jet');
colorbar;
title('Q-values (FDR corrected)');
xlabel('Conditions');
ylabel('Percentiles');
set(gca, 'XTick', 1:3, 'XTickLabel', {'Full', 'Partial', 'No'});
set(gca, 'YTick', 1:length(percentiles), 'YTickLabel', percentiles);
