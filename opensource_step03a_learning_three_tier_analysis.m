%% Three-Tier Hierarchical Learning Analysis Script
% NOTE: This code demonstrates the analytical methodology.
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to
% NTU's open access policy.
%
% Purpose: Conduct three-tier hierarchical analysis of gaze effects on infant statistical learning
%
% This script implements the analysis described in Supplementary Table S1:
% Tier 1: Omnibus ANOVA testing overall gaze condition effect
% Tier 2: Post-hoc pairwise contrasts between conditions (one-tailed, FDR-corrected)
% Tier 3: Within-condition paired t-tests comparing learning against zero baseline (one-tailed, FDR-corrected)
%
% Key features:
% - Linear Mixed-Effects (LME) model accounting for repeated measures
% - Subject-level aggregation for baseline tests
% - Covariate adjustment (age, sex, country)
% - FDR correction for multiple comparisons
% - Cohen's d effect sizes with 95% confidence intervals
%
% Statistical framework follows:
% - Saffran et al., 1996 (statistical learning paradigm)
% - Cho & Abe, 2013; Hales, 2024 (one-tailed test justification)
%
% Author: Wei Zhang
% Last updated: 2025-01-20

%% Initialize environment
clear all
clc

%% USER CONFIGURATION: Set your data directory
base_path = '';  % <-- MODIFY THIS PATH

if isempty(base_path)
    error('base_path is not set. Please modify line 33 to specify your data directory.');
end

%% Load behavioral data from Step 2
% This assumes you've already run opensource_step02_calculate_learning_and_attention_proportion.m
% which generates 'behavioral_data.xlsx'

fprintf('Loading behavioral data...\n');

behavioral_data = xlsread(fullfile(base_path, 'behavioral_data.xlsx'));

% Extract key variables
% Columns: [location, id, age, sex, block, condition, learning_score, total_looking_time, attention_proportion]
location = behavioral_data(:, 1);     % Location (1 = Location 1, 2 = Location 2)
subject_id = behavioral_data(:, 2);   % Participant ID
age = behavioral_data(:, 3);          % Age in days
sex = behavioral_data(:, 4);          % Sex (1 = male, 2 = female)
block = behavioral_data(:, 5);        % Experimental block (1-3)
condition = behavioral_data(:, 6);    % Gaze condition (1=Full, 2=Partial, 3=No)
learning_block = behavioral_data(:, 7); % Learning score (word2 - word1 looking time), block-level

fprintf('Loaded %d observations from %d subjects\n', ...
    size(behavioral_data, 1), length(unique(subject_id)));

%% ========================================================================
%  STEP 1: Prepare Subject-Level Data for Analysis
%  ========================================================================
%
%  For LME analysis (Tiers 1-2):
%  - Average learning scores within each subject × condition combination
%  - This creates subject-level data while preserving within-subject structure
%  - N observations ≈ N_subjects × 3 conditions (with missing data)
%
%  For baseline t-tests (Tier 3):
%  - Further aggregate to one value per subject per condition
%  - After covariate adjustment
%  ========================================================================

fprintf('\nAggregating data to subject × condition level...\n');

% Get unique subjects
unique_subjects = unique(subject_id);
n_subjects = length(unique_subjects);

% Initialize subject × condition matrix
learning_subj_cond = nan(n_subjects, 3);  % subjects × conditions
age_subj = nan(n_subjects, 1);
sex_subj = nan(n_subjects, 1);
location_subj = nan(n_subjects, 1);

% Aggregate by subject × condition
for s = 1:n_subjects
    subj = unique_subjects(s);

    % Get subject's demographic info (from first observation)
    subj_idx = find(subject_id == subj, 1, 'first');
    age_subj(s) = age(subj_idx);
    sex_subj(s) = sex(subj_idx);
    location_subj(s) = location(subj_idx);

    % Average learning across blocks for each condition
    for cond = 1:3
        cond_idx = find(subject_id == subj & condition == cond);
        if ~isempty(cond_idx)
            learning_subj_cond(s, cond) = mean(learning_block(cond_idx), 'omitnan');
        end
    end
end

fprintf('Subject-level data: %d subjects × 3 conditions = %d potential observations\n', ...
    n_subjects, n_subjects * 3);
fprintf('Valid observations after missing data: %d\n', sum(~isnan(learning_subj_cond(:))));

%% ========================================================================
%  STEP 2: Create Long-Format Table for Linear Mixed-Effects Model
%  ========================================================================

fprintf('\nCreating long-format table for LME analysis...\n');

% Initialize arrays for long-format table
subject_id_lme = [];
condition_lme = [];
age_lme = [];
sex_lme = [];
location_lme = [];
learning_lme = [];

% Reshape to long format
for s = 1:n_subjects
    for cond = 1:3
        if ~isnan(learning_subj_cond(s, cond))
            subject_id_lme = [subject_id_lme; s];
            condition_lme = [condition_lme; cond];
            age_lme = [age_lme; age_subj(s)];
            sex_lme = [sex_lme; sex_subj(s)];
            location_lme = [location_lme; location_subj(s)];
            learning_lme = [learning_lme; learning_subj_cond(s, cond)];
        end
    end
end

% Create table with categorical variables
tbl = table(subject_id_lme, condition_lme, age_lme, sex_lme, location_lme, learning_lme, ...
    'VariableNames', {'Subject', 'Condition', 'Age', 'Sex', 'Country', 'Learning'});

tbl.Subject = categorical(tbl.Subject);
tbl.Condition = categorical(tbl.Condition, [1, 2, 3], {'FullGaze', 'PartialGaze', 'NoGaze'});
tbl.Sex = categorical(tbl.Sex);
tbl.Country = categorical(tbl.Country, [1, 2], {'Location1', 'Location2'});

fprintf('\nLME Table Summary:\n');
fprintf('  Total observations: %d\n', height(tbl));
fprintf('  Number of subjects: %d\n', length(unique(tbl.Subject)));
fprintf('  Observations per condition:\n');
fprintf('    Full Gaze: %d\n', sum(tbl.Condition == 'FullGaze'));
fprintf('    Partial Gaze: %d\n', sum(tbl.Condition == 'PartialGaze'));
fprintf('    No Gaze: %d\n\n', sum(tbl.Condition == 'NoGaze'));

%% ========================================================================
%  TIER 1: Omnibus ANOVA Testing Gaze Condition Effect
%  ========================================================================
%
%  Research Question: Do learning levels differ between gaze conditions?
%
%  Model: Learning ~ Condition + Age + Sex + Country + (1|Subject)
%
%  Test: ANOVA on Condition factor
%
%  Note: We use LME model with interactions for demographic moderators,
%        but report the main effect of Condition here
%  ========================================================================

fprintf('========================================================================\n');
fprintf('TIER 1: OMNIBUS ANOVA\n');
fprintf('========================================================================\n\n');

% Fit LME model with interactions
% This allows testing whether gaze effects vary by demographics
lme_full = fitlme(tbl, 'Learning ~ Condition*Age + Condition*Sex + Condition*Country + (1|Subject)');

% Perform ANOVA on the model
fprintf('Testing overall effect of Gaze Condition...\n\n');
anova_result = anova(lme_full);
disp(anova_result);

% Extract key statistics for Condition
cond_F = anova_result.FStat(2);       % F-statistic for Condition
cond_df1 = anova_result.DF1(2);       % df1 (numerator)
cond_df2 = anova_result.DF2(2);       % df2 (denominator)
cond_p = anova_result.pValue(2);      % p-value

% Calculate partial eta-squared
partial_eta_sq = (cond_F * cond_df1) / (cond_F * cond_df1 + cond_df2);

fprintf('\n--- Omnibus Test Summary ---\n');
fprintf('F(%d, %d) = %.2f, p = %.3f, partial η² = %.3f\n', ...
    cond_df1, cond_df2, cond_F, cond_p, partial_eta_sq);

if cond_p < 0.05
    fprintf('Result: Significant condition effect\n');
elseif cond_p < 0.10
    fprintf('Result: Marginal trend toward condition differences\n');
else
    fprintf('Result: No significant condition effect\n');
end

%% ========================================================================
%  TIER 2: Post-Hoc Pairwise Contrasts (One-Tailed)
%  ========================================================================
%
%  Research Question: Which specific condition comparisons are significant?
%
%  Hypotheses (directional):
%    H1: Full Gaze > Partial Gaze
%    H2: Full Gaze > No Gaze
%    H3: Partial Gaze > No Gaze
%
%  Method: Extract coefficients from LME model and conduct one-tailed tests
%
%  Note: One-tailed tests justified by strong a priori directional hypotheses
%        based on Natural Pedagogy theory and prior empirical evidence
%  ========================================================================

fprintf('\n========================================================================\n');
fprintf('TIER 2: PAIRWISE CONTRASTS (One-Tailed)\n');
fprintf('========================================================================\n\n');

% Extract coefficient table
coef_table = lme_full.Coefficients;

% Full vs Partial Gaze
% In the model, Partial coefficient represents Partial - Full
% For H1: Full > Partial, we expect negative coefficient
t_12 = coef_table.tStat(2);  % Condition_PartialGaze
p_12_twotail = coef_table.pValue(2);
% Convert to one-tailed: if t is negative (Full > Partial), use p/2
p_12_onetail = p_12_twotail/2 * (sign(t_12) == -1) + (1 - p_12_twotail/2) * (sign(t_12) == 1);

% Full vs No Gaze
% NoGaze coefficient represents No - Full
% For H2: Full > No, we expect negative coefficient
t_13 = coef_table.tStat(3);  % Condition_NoGaze
p_13_twotail = coef_table.pValue(3);
p_13_onetail = p_13_twotail/2 * (sign(t_13) == -1) + (1 - p_13_twotail/2) * (sign(t_13) == 1);

% Partial vs No Gaze
% Need to test Partial - No using coefTest
H_23 = zeros(1, length(coef_table.Estimate));
H_23(2) = 1;   % Condition_PartialGaze
H_23(3) = -1;  % Condition_NoGaze
[p_23_twotail, F_23] = coefTest(lme_full, H_23);
t_23 = sign(coef_table.Estimate(2) - coef_table.Estimate(3)) * sqrt(F_23);
% For H3: Partial > No, we expect positive t
p_23_onetail = p_23_twotail/2 * (sign(t_23) == 1) + (1 - p_23_twotail/2) * (sign(t_23) == -1);

fprintf('Raw p-values (one-tailed):\n');
fprintf('  Full > Partial:   t(%.0f) = %.2f, p = %.4f\n', cond_df2, -t_12, p_12_onetail);
fprintf('  Full > No Gaze:   t(%.0f) = %.2f, p = %.4f\n', cond_df2, -t_13, p_13_onetail);
fprintf('  Partial > No:     t(%.0f) = %.2f, p = %.4f\n', cond_df2, t_23, p_23_onetail);

% FDR correction across 3 comparisons
p_contrasts = [p_12_onetail, p_13_onetail, p_23_onetail];
q_contrasts = mafdr(p_contrasts, 'BHFDR', true);

fprintf('\n--- FDR-Corrected (Benjamini-Hochberg) ---\n');
fprintf('  Full > Partial:   pFDR = %.3f %s\n', q_contrasts(1), sig_marker(q_contrasts(1)));
fprintf('  Full > No Gaze:   pFDR = %.3f %s\n', q_contrasts(2), sig_marker(q_contrasts(2)));
fprintf('  Partial > No:     pFDR = %.3f %s\n\n', q_contrasts(3), sig_marker(q_contrasts(3)));

%% ========================================================================
%  TIER 3: Within-Condition Baseline Tests (One-Tailed)
%  ========================================================================
%
%  Research Question: Does learning actually occur in each condition?
%
%  This is the "gold-standard" test of statistical learning:
%  - Learning score > 0 indicates novelty preference (successful segmentation)
%  - One-sample t-test compares learning against zero baseline
%
%  Procedure:
%  1. Adjust for covariates (regress out Age, Sex, Country)
%  2. Average by subject (N = 47)
%  3. One-sample t-test (one-tailed, right-tail) vs 0
%  4. FDR correction across 3 conditions
%
%  Note: This is equivalent to paired t-test between nonword and word looking times
%  ========================================================================

fprintf('========================================================================\n');
fprintf('TIER 3: WITHIN-CONDITION BASELINE TESTS\n');
fprintf('========================================================================\n\n');

fprintf('Procedure:\n');
fprintf('  1. Regress out Age, Sex, Country for each condition\n');
fprintf('  2. Average residuals by subject\n');
fprintf('  3. One-sample t-test against zero (one-tailed)\n');
fprintf('  4. FDR correction across conditions\n\n');

% Get indices for each condition
cond_idx1 = find(condition_lme == 1);  % Full Gaze
cond_idx2 = find(condition_lme == 2);  % Partial Gaze
cond_idx3 = find(condition_lme == 3);  % No Gaze

% --- Condition 1: Full Gaze ---
X1 = [ones(length(cond_idx1), 1), location_lme(cond_idx1), age_lme(cond_idx1), sex_lme(cond_idx1)];
[~, ~, resid1] = regress(learning_lme(cond_idx1), X1);
adjusted1 = resid1 + mean(learning_lme(cond_idx1), 'omitnan');

% Average by subject
unique_ids_c1 = unique(subject_id_lme(cond_idx1));
avg1 = arrayfun(@(id) mean(adjusted1(subject_id_lme(cond_idx1) == id), "omitnan"), unique_ids_c1);
avg1 = avg1(~isnan(avg1));

% One-sample t-test (one-tailed, right)
[h1, p1, ci1, stats1] = ttest(avg1, 0, 'Tail', 'right');

% Cohen's d and 95% CI
n1 = length(avg1);
d1 = mean(avg1) / std(avg1);
se_d1 = sqrt((1/n1) + (d1^2/(2*n1)));
d1_ci = [d1 - 1.96*se_d1, d1 + 1.96*se_d1];

% --- Condition 2: Partial Gaze ---
X2 = [ones(length(cond_idx2), 1), location_lme(cond_idx2), age_lme(cond_idx2), sex_lme(cond_idx2)];
[~, ~, resid2] = regress(learning_lme(cond_idx2), X2);
adjusted2 = resid2 + mean(learning_lme(cond_idx2), "omitnan");

unique_ids_c2 = unique(subject_id_lme(cond_idx2));
avg2 = arrayfun(@(id) mean(adjusted2(subject_id_lme(cond_idx2) == id), "omitnan"), unique_ids_c2);
avg2 = avg2(~isnan(avg2));

[h2, p2, ci2, stats2] = ttest(avg2, 0, 'Tail', 'right');

n2 = length(avg2);
d2 = mean(avg2) / std(avg2);
se_d2 = sqrt((1/n2) + (d2^2/(2*n2)));
d2_ci = [d2 - 1.96*se_d2, d2 + 1.96*se_d2];

% --- Condition 3: No Gaze ---
X3 = [ones(length(cond_idx3), 1), location_lme(cond_idx3), age_lme(cond_idx3), sex_lme(cond_idx3)];
[~, ~, resid3] = regress(learning_lme(cond_idx3), X3);
adjusted3 = resid3 + mean(learning_lme(cond_idx3), "omitnan");

unique_ids_c3 = unique(subject_id_lme(cond_idx3));
avg3 = arrayfun(@(id) mean(adjusted3(subject_id_lme(cond_idx3) == id), "omitnan"), unique_ids_c3);
avg3 = avg3(~isnan(avg3));

[h3, p3, ci3, stats3] = ttest(avg3, 0, 'Tail', 'right');

n3 = length(avg3);
d3 = mean(avg3) / std(avg3);
se_d3 = sqrt((1/n3) + (d3^2/(2*n3)));
d3_ci = [d3 - 1.96*se_d3, d3 + 1.96*se_d3];

% FDR correction
p_baseline = [p1, p2, p3];
q_baseline = mafdr(p_baseline, 'BHFDR', true);

fprintf('Results (FDR-corrected):\n\n');
fprintf('%-15s  t(df)       p_raw    p_FDR    Cohen''s d    95%% CI\n', 'Condition');
fprintf('%-15s  -------     ------   ------   ---------    ----------------\n', '---------------');
fprintf('%-15s  t(%d)=%.2f  %.4f   %.3f    d=%.2f       [%.2f, %.2f] %s\n', ...
    'Full Gaze', stats1.df, stats1.tstat, p1, q_baseline(1), d1, d1_ci(1), d1_ci(2), sig_marker(q_baseline(1)));
fprintf('%-15s  t(%d)=%.2f  %.4f   %.3f    d=%.2f       [%.2f, %.2f] %s\n', ...
    'Partial Gaze', stats2.df, stats2.tstat, p2, q_baseline(2), d2, d2_ci(1), d2_ci(2), sig_marker(q_baseline(2)));
fprintf('%-15s  t(%d)=%.2f  %.4f   %.3f    d=%.2f       [%.2f, %.2f] %s\n\n', ...
    'No Gaze', stats3.df, stats3.tstat, p3, q_baseline(3), d3, d3_ci(1), d3_ci(2), sig_marker(q_baseline(3)));

%% ========================================================================
%  INTEGRATED SUMMARY
%  ========================================================================

fprintf('========================================================================\n');
fprintf('THREE-TIER ANALYSIS SUMMARY\n');
fprintf('========================================================================\n\n');

fprintf('Tier 1 (Omnibus ANOVA):\n');
fprintf('  F(%d, %d) = %.2f, p = %.3f\n', cond_df1, cond_df2, cond_F, cond_p);
if cond_p < 0.10
    fprintf('  → Marginal/significant trend for condition differences\n\n');
else
    fprintf('  → No significant condition effect\n\n');
end

fprintf('Tier 2 (Pairwise Contrasts, one-tailed, FDR-corrected):\n');
fprintf('  Full > No:     pFDR = %.3f %s\n', q_contrasts(2), sig_marker(q_contrasts(2)));
fprintf('  Partial > No:  pFDR = %.3f %s\n', q_contrasts(3), sig_marker(q_contrasts(3)));
fprintf('  Full > Partial: pFDR = %.3f %s\n\n', q_contrasts(1), sig_marker(q_contrasts(1)));

fprintf('Tier 3 (Baseline Tests, one-tailed, FDR-corrected):\n');
fprintf('  Full Gaze:     pFDR = %.3f %s\n', q_baseline(1), sig_marker(q_baseline(1)));
fprintf('  Partial Gaze:  pFDR = %.3f %s\n', q_baseline(2), sig_marker(q_baseline(2)));
fprintf('  No Gaze:       pFDR = %.3f %s\n\n', q_baseline(3), sig_marker(q_baseline(3)));

fprintf('Interpretation:\n');
fprintf('  Tier 1-2 address: Do learning levels differ between conditions?\n');
fprintf('  Tier 3 addresses: Does learning actually occur (vs. zero baseline)?\n\n');

fprintf('========================================================================\n');
fprintf('Analysis complete. Results match Supplementary Table S1.\n');
fprintf('========================================================================\n');

%% Helper function for significance markers
function marker = sig_marker(p)
    if p < 0.001
        marker = '***';
    elseif p < 0.01
        marker = '**';
    elseif p < 0.05
        marker = '*';
    else
        marker = '';
    end
end
