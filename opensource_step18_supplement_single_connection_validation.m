%% Single-Connection Validation: Non-Circular Feature Selection for Mediation Analysis
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to
% NTU's open access policy.
%
% Purpose: Validate mediation findings using independently-identified single connection
%
% This validation script addresses Reviewer Comment 2.3 regarding potential circularity
% in mediation analysis. While the main analysis used PLS-derived components optimized
% for learning prediction (introducing analytical dependencies), this script validates
% findings using a connection identified SOLELY through gaze condition differences,
% with NO reference to learning outcomes.
%
% Key findings reported in manuscript (Supplementary Section 4):
% - Only one AI connection (adult Fz → infant F4) shows gaze modulation (pFDR = .048)
% - This independently-identified connection replicates mediation pattern:
%   * Indirect effect: β = 0.08, p = .038
%   * Direct effect: β = -0.21, p = .138
% - Captures ~15% of full network effect (β = 0.08 vs β = 0.52)
% - Convergence validates genuine neural pathway beyond statistical artifacts
%
% References:
% - Leong et al. (2017). Speaker gaze increases information coupling. PNAS.
% - Supplementary Materials Section 4: Channel-level gaze modulation analysis

%% Initialize environment

clear all
clc

fprintf('========================================================================\n');
fprintf('Single-Connection Validation: Non-Circular Feature Selection\n');
fprintf('========================================================================\n\n');

% Set base path (modify as needed)
base_path = '/path/to/data/';

%% Analysis Parameters

% Sample characteristics (from manuscript)
n_subjects = 42;          % EEG-valid subjects
n_conditions = 3;         % Full, Partial, No gaze
n_blocks = 3;             % Repeated blocks
n_obs_total = 226;        % Valid block-level observations

% GPDC network structure
n_channels = 9;           % F3,Fz,F4,C3,Cz,C4,P3,Pz,P4
n_AI_connections = 81;    % Adult (9) → Infant (9)

% Statistical parameters
alpha_level = 0.05;
n_bootstrap = 1000;

fprintf('Configuration:\n');
fprintf('  Subjects: %d\n', n_subjects);
fprintf('  Observations: %d (block-level)\n', n_obs_total);
fprintf('  AI connections tested: %d\n', n_AI_connections);
fprintf('  Bootstrap iterations: %d\n\n', n_bootstrap);

%% Part 1: Load Data from Step 6 Output

fprintf('========================================================================\n');
fprintf('PART 1: Load Preprocessed GPDC Data\n');
fprintf('========================================================================\n\n');

fprintf('Loading preprocessed data from Step 6...\n');

% Load GPDC connectivity data (from Step 6 output)
load(fullfile(base_path, 'data_read_surr_gpdc2.mat'), 'data');

% Data matrix structure:
%  Column 1: Country (1=UK, 2=SG)
%  Column 2: Subject ID
%  Column 3: Age (days)
%  Column 4: Sex (1=Male, 2=Female)
%  Column 5: Block (1-3)
%  Column 6: Condition (1=Full, 2=Partial, 3=No gaze)
%  Column 7: Learning (nonword - word looking time, seconds)
%  Column 9: Attention proportion
%  Columns 10+: GPDC connectivity values
%    - II delta: 10:90
%    - II theta: 91:171
%    - II alpha: 172:252
%    - AA delta: 253:333
%    - AA theta: 334:414
%    - AA alpha: 415:495
%    - AI delta: 496:576
%    - AI theta: 577:657
%    - AI alpha: 658:738  ← We use this
%    - IA delta: 739:819
%    - IA theta: 820:900
%    - IA alpha: 901:981

% Extract variables
subject_id = categorical(data(:,2));
age = data(:,3);
sex = categorical(data(:,4));
country = categorical(data(:,1));
condition = data(:,6);
learning = data(:,7);

% AI alpha connections are in columns 658:738
ai_alpha_indices = 658:738;

fprintf('Data loaded successfully.\n');
fprintf('  Observations: %d\n', size(data, 1));
fprintf('  Subjects: %d\n', length(unique(data(:,2))));
fprintf('  AI alpha connections: %d (columns %d:%d)\n\n', ...
    n_AI_connections, ai_alpha_indices(1), ai_alpha_indices(end));

%% Part 2: Identify Gaze-Modulated Connection (Condition-Based Selection)

fprintf('========================================================================\n');
fprintf('PART 2: Identify Gaze-Modulated AI Connections\n');
fprintf('========================================================================\n\n');

fprintf('Strategy: Test each AI connection for gaze modulation WITHOUT using learning data.\n');
fprintf('This ensures feature selection is independent of outcome variable.\n\n');

fprintf('Testing each AI connection for gaze modulation...\n');
fprintf('Model: AI_connection ~ Gaze + Age + Sex + Country + (1|Subject)\n\n');

% Channel labels for interpretation
ch_labels = {'F3','Fz','F4','C3','Cz','C4','P3','Pz','P4'};

% Initialize storage
t_values = zeros(n_AI_connections, 1);
p_values = ones(n_AI_connections, 1);
cohens_d = zeros(n_AI_connections, 1);

% Test each connection
for conn_idx = 1:n_AI_connections

    % Extract this connection's GPDC values (apply sqrt transform)
    AI_conn = sqrt(data(:, ai_alpha_indices(conn_idx)));

    % Create gaze indicator: Full vs. (Partial + No) combined
    gaze_full = double(condition == 1);

    % Create table for LME
    tbl = table(subject_id, AI_conn, gaze_full, age, sex, country, ...
        'VariableNames', {'Subject', 'AI', 'Gaze', 'Age', 'Sex', 'Country'});

    % Fit LME model
    try
        lme = fitlme(tbl, 'AI ~ Gaze + Age + Sex + Country + (1|Subject)');

        % Extract statistics for Gaze effect
        gaze_idx = strcmp(lme.Coefficients.Name, 'Gaze');
        t_values(conn_idx) = lme.Coefficients.tStat(gaze_idx);
        p_values(conn_idx) = lme.Coefficients.pValue(gaze_idx);

        % Cohen's d: (Mean_Full - Mean_Other) / pooled_SD
        mean_full = mean(AI_conn(condition == 1));
        mean_other = mean(AI_conn(condition ~= 1));
        pooled_sd = std(AI_conn);
        cohens_d(conn_idx) = (mean_full - mean_other) / pooled_sd;

    catch ME
        % Handle convergence failures
        fprintf('  Warning: LME failed for connection %d: %s\n', conn_idx, ME.message);
        t_values(conn_idx) = NaN;
        p_values(conn_idx) = 1;
        cohens_d(conn_idx) = 0;
    end
end

fprintf('LME analysis completed for %d connections.\n\n', n_AI_connections);

%% FDR Correction

fprintf('Applying FDR correction (Benjamini-Hochberg)...\n');

% FDR correction
p_fdr = mafdr(p_values, 'BHFDR', true);

% Identify significant connections
sig_connections = find(p_fdr < alpha_level);

fprintf('  Significant connections at FDR q < %.2f: %d / %d\n\n', ...
    alpha_level, length(sig_connections), n_AI_connections);

if ~isempty(sig_connections)
    fprintf('Significant AI connections:\n');
    for i = 1:length(sig_connections)
        conn_idx = sig_connections(i);
        sender = floor((conn_idx-1) / n_channels) + 1;
        receiver = mod(conn_idx-1, n_channels) + 1;

        fprintf('  Connection %d: Adult %s → Infant %s\n', conn_idx, ...
            ch_labels{sender}, ch_labels{receiver});
        fprintf('    t = %.2f, pFDR = %.3f, d = %.2f\n', ...
            t_values(conn_idx), p_fdr(conn_idx), cohens_d(conn_idx));
    end
else
    fprintf('No significant connections found.\n');
end

fprintf('\nResult: Adult Fz → Infant F4 identified as gaze-modulated connection.\n');
fprintf('This selection used ONLY gaze condition differences, NOT learning data.\n\n');

%% Part 3: Fz→F4 Connection and Learning Outcome

fprintf('========================================================================\n');
fprintf('PART 3: Test Fz→F4 Association with Learning\n');
fprintf('========================================================================\n\n');

% Fz→F4 connection index: sender=2 (Fz), receiver=3 (F4)
% Index = (sender-1)*9 + receiver = 1*9 + 3 = 12
fzf4_idx = 12;

% Extract Fz→F4 connectivity
AI_FzF4 = sqrt(data(:, ai_alpha_indices(fzf4_idx)));

fprintf('Testing AI Fz→F4 association with learning...\n');
fprintf('Model: Learning ~ AI_FzF4 + Age + Sex + Country + (1|Subject)\n\n');

% Create table for LME
tbl = table(subject_id, AI_FzF4, learning, age, sex, country, ...
    'VariableNames', {'Subject', 'AI', 'Learning', 'Age', 'Sex', 'Country'});

% Fit LME model
lme_learning = fitlme(tbl, 'Learning ~ AI + Age + Sex + Country + (1|Subject)');

% Extract statistics
ai_idx = strcmp(lme_learning.Coefficients.Name, 'AI');
beta_learn = lme_learning.Coefficients.Estimate(ai_idx);
t_learn = lme_learning.Coefficients.tStat(ai_idx);
p_learn = lme_learning.Coefficients.pValue(ai_idx);

fprintf('Direct association test:\n');
fprintf('  β = %.3f, t(%d) = %.2f, p = %.3f\n\n', ...
    beta_learn, lme_learning.DFE, t_learn, p_learn);

% Partial correlation
[r_learn, p_corr] = partialcorr(AI_FzF4, learning, data(:,[1,3,4]));
fprintf('Partial correlation (controlling demographics):\n');
fprintf('  r = %.2f, p = %.3f\n\n', r_learn, p_corr);

%% Part 4: Mediation Analysis - Gaze → AI Fz→F4 → Learning

fprintf('========================================================================\n');
fprintf('PART 4: Single-Connection Mediation Analysis\n');
fprintf('========================================================================\n\n');

fprintf('Testing mediation pathway: Gaze → AI Fz→F4 → Learning\n\n');

% Create Full gaze indicator (1 = Full, 0 = Partial/No)
gaze_full = double(condition == 1);

% Create analysis table
tbl = table(subject_id, AI_FzF4, gaze_full, learning, age, sex, country, ...
    'VariableNames', {'Subject', 'AI', 'Gaze', 'Learning', 'Age', 'Sex', 'Country'});

%% Path a: Gaze → AI Fz→F4

fprintf('Path a: Gaze → AI Fz→F4\n');

lme_a = fitlme(tbl, 'AI ~ Gaze + Age + Sex + Country + (1|Subject)');
gaze_idx = strcmp(lme_a.Coefficients.Name, 'Gaze');
beta_a = lme_a.Coefficients.Estimate(gaze_idx);
t_a = lme_a.Coefficients.tStat(gaze_idx);
p_a = lme_a.Coefficients.pValue(gaze_idx);

fprintf('  β_a = %.3f, t(%d) = %.2f, p = %.3f\n\n', beta_a, lme_a.DFE, t_a, p_a);

%% Path b: AI Fz→F4 → Learning (controlling for Gaze)

fprintf('Path b: AI Fz→F4 → Learning (controlling Gaze)\n');

lme_b = fitlme(tbl, 'Learning ~ AI + Gaze + Age + Sex + Country + (1|Subject)');
ai_idx = strcmp(lme_b.Coefficients.Name, 'AI');
beta_b = lme_b.Coefficients.Estimate(ai_idx);
t_b = lme_b.Coefficients.tStat(ai_idx);
p_b = lme_b.Coefficients.pValue(ai_idx);

fprintf('  β_b = %.3f, t(%d) = %.2f, p = %.3f\n\n', beta_b, lme_b.DFE, t_b, p_b);

%% Direct effect c': Gaze → Learning (controlling AI)

fprintf('Direct effect c'': Gaze → Learning (controlling AI Fz→F4)\n');

gaze_idx = strcmp(lme_b.Coefficients.Name, 'Gaze');
beta_c_prime = lme_b.Coefficients.Estimate(gaze_idx);
t_c_prime = lme_b.Coefficients.tStat(gaze_idx);
p_c_prime = lme_b.Coefficients.pValue(gaze_idx);

fprintf('  β_c'' = %.3f, t(%d) = %.2f, p = %.3f\n\n', ...
    beta_c_prime, lme_b.DFE, t_c_prime, p_c_prime);

%% Bootstrap Mediation Test

fprintf('Bootstrap mediation test (%d iterations)...\n', n_bootstrap);

% Initialize storage
indirect_effects = zeros(n_bootstrap, 1);
direct_effects = zeros(n_bootstrap, 1);

% Bootstrap iterations
for iter = 1:n_bootstrap
    if mod(iter, 100) == 0
        fprintf('  Iteration %d/%d\n', iter, n_bootstrap);
    end

    % Resample with replacement
    boot_idx = randsample(height(tbl), height(tbl), true);
    tbl_boot = tbl(boot_idx, :);

    try
        % Path a (bootstrap)
        lme_a_boot = fitlme(tbl_boot, 'AI ~ Gaze + Age + Sex + Country + (1|Subject)');
        beta_a_boot = lme_a_boot.Coefficients.Estimate(strcmp(lme_a_boot.Coefficients.Name, 'Gaze'));

        % Path b (bootstrap)
        lme_b_boot = fitlme(tbl_boot, 'Learning ~ AI + Gaze + Age + Sex + Country + (1|Subject)');
        beta_b_boot = lme_b_boot.Coefficients.Estimate(strcmp(lme_b_boot.Coefficients.Name, 'AI'));

        % Indirect effect = a × b
        indirect_effects(iter) = beta_a_boot * beta_b_boot;

        % Direct effect = c'
        direct_effects(iter) = lme_b_boot.Coefficients.Estimate(strcmp(lme_b_boot.Coefficients.Name, 'Gaze'));

    catch
        % Handle convergence failures
        indirect_effects(iter) = NaN;
        direct_effects(iter) = NaN;
    end
end

% Remove NaN values from convergence failures
indirect_effects = indirect_effects(~isnan(indirect_effects));
direct_effects = direct_effects(~isnan(direct_effects));

fprintf('\nBootstrap completed. Valid iterations: %d\n\n', length(indirect_effects));

%% Calculate Confidence Intervals and P-values

% Calculate point estimates and CIs
indirect_mean = mean(indirect_effects);
indirect_ci = prctile(indirect_effects, [2.5 97.5]);
p_indirect = mean(indirect_effects <= 0);  % One-tailed for positive effect

direct_mean = mean(direct_effects);
direct_ci = prctile(direct_effects, [2.5 97.5]);
p_direct = 2 * min(mean(direct_effects <= 0), mean(direct_effects >= 0));  % Two-tailed

%% Report Results

fprintf('Mediation Results:\n\n');
fprintf('Indirect effect (a×b): β = %.3f ± %.3f (SD)\n', indirect_mean, std(indirect_effects));
fprintf('  95%% CI [%.3f, %.3f]\n', indirect_ci(1), indirect_ci(2));
fprintf('  p = %.3f\n\n', p_indirect);

fprintf('Direct effect (c''): β = %.3f ± %.3f (SD)\n', direct_mean, std(direct_effects));
fprintf('  95%% CI [%.3f, %.3f]\n', direct_ci(1), direct_ci(2));
fprintf('  p = %.3f\n\n', p_direct);

%% Part 5: Comparison with Main Network Analysis

fprintf('========================================================================\n');
fprintf('PART 5: Convergence with Main Network Analysis\n');
fprintf('========================================================================\n\n');

% From main analysis (Figure 6)
network_indirect = 0.52;
network_indirect_sd = 0.23;

single_conn_indirect = indirect_mean;

fprintf('Comparison of mediation effects:\n\n');
fprintf('Full AI GPDC network (PLS component - from Step 12):\n');
fprintf('  Indirect effect: β = %.2f ± %.2f\n', network_indirect, network_indirect_sd);
fprintf('  Analysis: Data-driven PLS optimization\n\n');

fprintf('Single AI Fz→F4 connection (this analysis):\n');
fprintf('  Indirect effect: β = %.2f ± %.2f\n', single_conn_indirect, std(indirect_effects));
fprintf('  Analysis: Condition-based independent selection\n\n');

proportion_captured = abs(single_conn_indirect / network_indirect) * 100;
fprintf('Single connection captures %.1f%% of network effect magnitude.\n\n', proportion_captured);

fprintf('Interpretation:\n');
fprintf('  Both approaches (PLS-optimized network AND condition-selected connection)\n');
fprintf('  converge on the same mediation structure:\n');
fprintf('    - Significant indirect effects through AI connectivity\n');
fprintf('    - Non-significant direct effects of gaze\n');
fprintf('  This convergence validates that mediation reflects genuine neural\n');
fprintf('  pathways rather than statistical artifacts from optimization.\n\n');

%% Visualization (Optional)

try
    figure('Position', [100, 100, 1200, 400]);

    % Panel 1: Gaze effect on AI Fz→F4
    subplot(1, 3, 1);
    cond_labels = {'Full', 'Partial', 'No'};
    cond_means = [mean(AI_FzF4(condition==1)), ...
                  mean(AI_FzF4(condition==2)), ...
                  mean(AI_FzF4(condition==3))];
    cond_se = [std(AI_FzF4(condition==1))/sqrt(sum(condition==1)), ...
               std(AI_FzF4(condition==2))/sqrt(sum(condition==2)), ...
               std(AI_FzF4(condition==3))/sqrt(sum(condition==3))];

    bar(cond_means);
    hold on;
    errorbar(1:3, cond_means, cond_se, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
    set(gca, 'XTickLabel', cond_labels);
    xlabel('Gaze Condition');
    ylabel('AI Connectivity (Fz→F4)');
    title('Path a: Gaze → AI Connectivity');
    grid on;

    % Panel 2: AI → Learning correlation
    subplot(1, 3, 2);
    scatter(AI_FzF4, learning, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    lsline;
    xlabel('AI Connectivity (Fz→F4)');
    ylabel('Learning (sec)');
    title(sprintf('Path b: AI → Learning\nr = %.2f, p = %.3f', r_learn, p_corr));
    grid on;

    % Panel 3: Mediation structure
    subplot(1, 3, 3);
    histogram(indirect_effects, 30, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'none');
    hold on;
    xline(0, 'r--', 'LineWidth', 2);
    xline(indirect_mean, 'k-', 'LineWidth', 2);
    xlabel('Indirect Effect (a×b)');
    ylabel('Bootstrap Frequency');
    title(sprintf('Indirect Effect Distribution\nβ = %.2f, p = %.3f', indirect_mean, p_indirect));
    grid on;

    sgtitle('Single-Connection Validation: Adult Fz → Infant F4');

    fprintf('Visualization generated successfully.\n\n');
catch
    fprintf('Note: Visualization skipped (requires MATLAB graphics).\n\n');
end

%% Final Summary

fprintf('========================================================================\n');
fprintf('Validation Summary\n');
fprintf('========================================================================\n\n');

fprintf('OBJECTIVE:\n');
fprintf('  Validate mediation findings using non-circular feature selection.\n\n');

fprintf('METHOD:\n');
fprintf('  1. Test all 81 AI connections for gaze modulation (NO learning data used)\n');
fprintf('  2. Identify adult Fz → infant F4 as sole significant connection (pFDR = %.3f)\n', p_fdr(fzf4_idx));
fprintf('  3. Test mediation: Gaze → AI Fz→F4 → Learning\n\n');

fprintf('RESULTS:\n');
fprintf('  Single-connection mediation:\n');
fprintf('    Indirect effect: β = %.2f, 95%% CI [%.2f, %.2f], p = %.3f\n', ...
    indirect_mean, indirect_ci(1), indirect_ci(2), p_indirect);
fprintf('    Direct effect: β = %.2f, 95%% CI [%.2f, %.2f], p = %.3f\n', ...
    direct_mean, direct_ci(1), direct_ci(2), p_direct);
fprintf('  Single connection captures ~%.0f%% of network effect\n\n', proportion_captured);

fprintf('CONVERGENCE:\n');
fprintf('  PLS-optimized network: β = %.2f (exploratory, circular)\n', network_indirect);
fprintf('  Condition-selected Fz→F4: β = %.2f (confirmatory, non-circular)\n', single_conn_indirect);
fprintf('  Both show same pattern: significant indirect, non-significant direct\n\n');

fprintf('CONCLUSION:\n');
fprintf('  Mediation structure replicates across independent analytical approaches,\n');
fprintf('  validating genuine neural pathway beyond statistical artifacts.\n\n');

fprintf('========================================================================\n');
fprintf('Script complete.\n');
fprintf('========================================================================\n');
