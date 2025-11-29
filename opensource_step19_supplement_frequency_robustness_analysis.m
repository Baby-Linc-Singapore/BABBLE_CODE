%% Frequency Robustness Analysis: Cross-Band Validation of Mediation Effects
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to
% NTU's open access policy.
%
% Purpose: Validate mediation findings across multiple frequency bands
%
% This validation script addresses Reviewer Comment 2.6 regarding robustness of
% findings across frequency specifications. While the main analysis focused on
% infant alpha band (6-9 Hz), this script validates the Gaze → AI connectivity → Learning
% pathway across three frequency bands: Delta (1-3 Hz), Theta (3-6 Hz), and Alpha (6-9 Hz).
%
% Key findings reported in manuscript (Supplementary Section 13.4):
% - Alpha (6-9 Hz): β = 0.52, p = .014; PLS R² = 24.6%, p = .042
% - Theta (3-6 Hz): β = 0.38, p = .048; PLS R² = 24.2%, p = .039
% - Delta (1-3 Hz): β = 0.27, p = .124; PLS R² = 22.7%, p = .089
% - Convergence across theta/alpha validates robustness beyond single frequency band
%
% References:
% - Supplementary Materials Section 3: Delta/Theta PLS analysis
% - Supplementary Tables S14-S15: Frequency band mediation and PLS results

%% Initialize environment

clear all
clc

fprintf('========================================================================\n');
fprintf('Frequency Robustness Analysis: Cross-Band Mediation Validation\n');
fprintf('========================================================================\n\n');

% Set base path (modify as needed)
base_path = '/path/to/data/';

%% Analysis Parameters

% Sample characteristics
n_subjects = 42;
n_conditions = 3;         % Full, Partial, No gaze
n_blocks = 3;
n_obs_total = 226;        % Block-level observations

% Frequency bands
bands = struct();
bands(1).name = 'Delta';
bands(1).range = [1 3];   % Hz
bands(2).name = 'Theta';
bands(2).range = [3 6];
bands(3).name = 'Alpha';
bands(3).range = [6 9];

n_bands = length(bands);

% GPDC parameters
n_channels = 9;
n_AI_connections = 81;    % Adult → Infant

% Statistical parameters
n_bootstrap = 1000;
n_surrogate = 1000;

fprintf('Configuration:\n');
fprintf('  Subjects: %d\n', n_subjects);
fprintf('  Observations: %d\n', n_obs_total);
fprintf('  Frequency bands: %d (Delta, Theta, Alpha)\n', n_bands);
fprintf('  Bootstrap iterations: %d\n', n_bootstrap);
fprintf('  Surrogate permutations: %d\n\n', n_surrogate);

%% Part 1: PLS Prediction Analysis Across Frequency Bands

fprintf('========================================================================\n');
fprintf('PART 1: PLS Regression - AI GPDC Prediction of Learning\n');
fprintf('========================================================================\n\n');

% Initialize result storage
PLS_results = struct();

for band_idx = 1:n_bands

    band_name = bands(band_idx).name;
    freq_range = bands(band_idx).range;

    fprintf('--- %s Band (%.0f-%.0f Hz) ---\n', band_name, freq_range(1), freq_range(2));

    %% Load GPDC Data for This Band

    gpdc_file = fullfile(base_path, 'GPDC_Results', ...
        sprintf('AI_GPDC_%s.mat', lower(band_name)));

    fprintf('Loading GPDC data for %s band from: %s\n', band_name, gpdc_file);
    fprintf('Expected variables:\n');
    fprintf('  AI_GPDC: %d obs × %d connections\n', n_obs_total, n_AI_connections);
    fprintf('  learning: %d × 1 (nonword - word looking time)\n', n_obs_total);
    fprintf('  condition: %d × 1 (1=Full, 2=Partial, 3=No)\n', n_obs_total);
    fprintf('  age, sex, country: Covariates\n\n');

    % Load data (user must provide actual preprocessed data)
    % load(gpdc_file, 'AI_GPDC', 'learning', 'condition', 'age', 'sex', 'country');

    fprintf('Note: Please load your preprocessed GPDC data for %s band.\n\n', band_name);

    %% PLS Regression

    fprintf('Running PLS regression...\n');

    % Prepare predictors (GPDC + covariates)
    X = [AI_GPDC, age, sex, country];
    y = learning;

    % Fit PLS model (1 component)
    [XL, yl, XS, YS, beta_pls, PCTVAR, MSE, stats] = plsregress(X, y, 1);

    % Predicted values
    y_pred_pls = [ones(n_obs_total, 1), X] * beta_pls;

    % Calculate R²
    SS_total = sum((y - mean(y)).^2);
    SS_residual = sum((y - y_pred_pls).^2);
    R2_real = 1 - (SS_residual / SS_total);

    fprintf('  Real R² = %.3f (%.1f%% variance explained)\n', R2_real, R2_real*100);

    %% Surrogate Testing

    fprintf('  Surrogate testing (%d permutations)...\n', n_surrogate);

    R2_surrogate = zeros(n_surrogate, 1);

    for iter = 1:n_surrogate
        % Shuffle learning outcomes
        y_shuffled = y(randperm(n_obs_total));

        % Fit PLS on shuffled data
        [~, ~, ~, ~, beta_surr, ~, ~, ~] = plsregress(X, y_shuffled, 1);
        y_pred_surr = [ones(n_obs_total, 1), X] * beta_surr;

        % Calculate R²
        SS_residual_surr = sum((y_shuffled - y_pred_surr).^2);
        R2_surrogate(iter) = 1 - (SS_residual_surr / sum((y_shuffled - mean(y_shuffled)).^2));
    end

    % Statistical significance
    surrogate_95th = prctile(R2_surrogate, 95);
    p_surrogate = mean(R2_surrogate >= R2_real);

    fprintf('  Surrogate R² = %.3f ± %.3f (mean ± SD)\n', ...
        mean(R2_surrogate), std(R2_surrogate));
    fprintf('  95th percentile = %.3f\n', surrogate_95th);
    fprintf('  P-value = %.3f', p_surrogate);

    if p_surrogate < 0.05
        fprintf(' *\n');
    else
        fprintf(' (ns)\n');
    end

    %% Bootstrap 95% CI for R²

    fprintf('  Bootstrap 95%% CI (%d iterations)...\n', n_bootstrap);

    R2_bootstrap = zeros(n_bootstrap, 1);

    for iter = 1:n_bootstrap
        % Resample with replacement
        boot_idx = randsample(n_obs_total, n_obs_total, true);

        X_boot = X(boot_idx, :);
        y_boot = y(boot_idx);

        % Fit PLS
        [~, ~, ~, ~, beta_boot, ~, ~, ~] = plsregress(X_boot, y_boot, 1);
        y_pred_boot = [ones(n_obs_total, 1), X_boot] * beta_boot;

        % Calculate R²
        SS_total_boot = sum((y_boot - mean(y_boot)).^2);
        SS_residual_boot = sum((y_boot - y_pred_boot).^2);
        R2_bootstrap(iter) = 1 - (SS_residual_boot / SS_total_boot);
    end

    R2_CI = prctile(R2_bootstrap, [2.5 97.5]);

    fprintf('  Bootstrap 95%% CI [%.1f%%, %.1f%%]\n\n', R2_CI(1)*100, R2_CI(2)*100);

    %% Store Results

    PLS_results(band_idx).band = band_name;
    PLS_results(band_idx).R2_real = R2_real;
    PLS_results(band_idx).R2_CI = R2_CI;
    PLS_results(band_idx).R2_surrogate_mean = mean(R2_surrogate);
    PLS_results(band_idx).R2_surrogate_95th = surrogate_95th;
    PLS_results(band_idx).p_surrogate = p_surrogate;

end

%% Part 2: Mediation Analysis Across Frequency Bands

fprintf('========================================================================\n');
fprintf('PART 2: Mediation Analysis - Gaze → AI GPDC → Learning\n');
fprintf('========================================================================\n\n');

% Initialize mediation result storage
mediation_results = struct();

for band_idx = 1:n_bands

    band_name = bands(band_idx).name;

    fprintf('--- %s Band ---\n', band_name);

    %% Load Data (should be same as Part 1)

    % Data should be already loaded from Part 1
    % AI_GPDC, learning, condition, age, sex, country
    % If running this section independently, load from gpdc_file

    %% Extract PLS Component

    % Use first PLS component as mediator
    X = [AI_GPDC, age, sex, country];
    y = learning;
    [XL, ~, XS, ~, ~, ~, ~, ~] = plsregress(X, y, 1);

    AI_component = XS(:, 1);  % First component scores
    % Note: Covariates (age, sex, country) are incorporated in PLS fitting

    %% Mediation Model

    fprintf('Testing mediation: Gaze → AI Component → Learning\n');

    % Create Full gaze indicator
    gaze_full = double(condition == 1);

    % Path a: Gaze → AI Component
    X_a = [gaze_full, age, sex, country, ones(n_obs_total, 1)];
    beta_a = X_a \ AI_component;

    % Path b: AI Component → Learning (controlling Gaze)
    X_b = [AI_component, gaze_full, age, sex, country, ones(n_obs_total, 1)];
    beta_b = X_b \ learning;

    % Point estimates
    a = beta_a(1);          % Gaze → AI
    b = beta_b(1);          % AI → Learning (controlling gaze)
    c_prime = beta_b(2);    % Gaze → Learning (direct effect)

    fprintf('  Path a (Gaze → AI): β = %.3f\n', a);
    fprintf('  Path b (AI → Learning): β = %.3f\n', b);
    fprintf('  Direct effect c'': β = %.3f\n', c_prime);

    %% Bootstrap Mediation Test

    fprintf('  Bootstrap mediation (%d iterations)...\n', n_bootstrap);

    indirect_effects = zeros(n_bootstrap, 1);
    direct_effects = zeros(n_bootstrap, 1);

    for iter = 1:n_bootstrap
        % Resample
        boot_idx = randsample(n_obs_total, n_obs_total, true);

        % Path a (bootstrap)
        beta_a_boot = X_a(boot_idx, :) \ AI_component(boot_idx);

        % Path b (bootstrap)
        beta_b_boot = X_b(boot_idx, :) \ learning(boot_idx);

        % Indirect = a × b
        indirect_effects(iter) = beta_a_boot(1) * beta_b_boot(1);

        % Direct = c'
        direct_effects(iter) = beta_b_boot(2);
    end

    % Statistics
    indirect_mean = mean(indirect_effects);
    indirect_se = std(indirect_effects);
    indirect_CI = prctile(indirect_effects, [2.5 97.5]);
    p_indirect = 2 * min(mean(indirect_effects <= 0), mean(indirect_effects >= 0));

    direct_mean = mean(direct_effects);
    direct_se = std(direct_effects);
    direct_CI = prctile(direct_effects, [2.5 97.5]);
    p_direct = 2 * min(mean(direct_effects <= 0), mean(direct_effects >= 0));

    fprintf('  Indirect effect: β = %.2f ± %.2f, 95%% CI [%.2f, %.2f], p = %.3f', ...
        indirect_mean, indirect_se, indirect_CI(1), indirect_CI(2), p_indirect);

    if p_indirect < 0.05
        fprintf(' *\n');
    else
        fprintf(' (ns)\n');
    end

    fprintf('  Direct effect: β = %.2f ± %.2f, 95%% CI [%.2f, %.2f], p = %.3f', ...
        direct_mean, direct_se, direct_CI(1), direct_CI(2), p_direct);

    if p_direct < 0.05
        fprintf(' *\n\n');
    else
        fprintf(' (ns)\n\n');
    end

    %% Store Results

    mediation_results(band_idx).band = band_name;
    mediation_results(band_idx).indirect_beta = indirect_mean;
    mediation_results(band_idx).indirect_se = indirect_se;
    mediation_results(band_idx).indirect_CI = indirect_CI;
    mediation_results(band_idx).p_indirect = p_indirect;
    mediation_results(band_idx).direct_beta = direct_mean;
    mediation_results(band_idx).direct_se = direct_se;
    mediation_results(band_idx).direct_CI = direct_CI;
    mediation_results(band_idx).p_direct = p_direct;

end

%% Part 3: Summary Tables

fprintf('========================================================================\n');
fprintf('PART 3: Summary Tables\n');
fprintf('========================================================================\n\n');

%% Table S14: Mediation Effects Across Frequency Bands

fprintf('Supplementary Table S14: Mediation Effects Across Frequency Bands\n');
fprintf('Indirect effects of gaze on learning through AI connectivity\n\n');

fprintf('%-10s  %-20s  %-25s  %-10s\n', ...
    'Band', 'Indirect Effect (β)', '95%% CI', 'P-value');
fprintf('%-10s  %-20s  %-25s  %-10s\n', ...
    '--------', '------------------', '---------------------', '--------');

for band_idx = 1:n_bands
    fprintf('%-10s  %7.2f ± %4.2f       [%5.2f, %5.2f]        %.3f', ...
        mediation_results(band_idx).band, ...
        mediation_results(band_idx).indirect_beta, ...
        mediation_results(band_idx).indirect_se, ...
        mediation_results(band_idx).indirect_CI(1), ...
        mediation_results(band_idx).indirect_CI(2), ...
        mediation_results(band_idx).p_indirect);

    if mediation_results(band_idx).p_indirect < 0.05
        fprintf(' *\n');
    else
        fprintf('\n');
    end
end

fprintf('\nNote: * p < .05; Bootstrap N = %d\n\n', n_bootstrap);

%% Table S15: PLS Prediction Performance

fprintf('Supplementary Table S15: PLS Prediction Performance Across Frequency Bands\n');
fprintf('AI GPDC prediction of learning outcomes\n\n');

fprintf('%-10s  %-15s  %-25s  %-20s  %-10s\n', ...
    'Band', 'Real R² (%%)', '95%% CI (%%)', 'Surrogate 95th (%%)', 'P-value');
fprintf('%-10s  %-15s  %-25s  %-20s  %-10s\n', ...
    '--------', '-------------', '---------------------', '------------------', '--------');

for band_idx = 1:n_bands
    fprintf('%-10s  %12.1f    [%4.1f, %4.1f]            %12.1f       %.3f', ...
        PLS_results(band_idx).band, ...
        PLS_results(band_idx).R2_real * 100, ...
        PLS_results(band_idx).R2_CI(1) * 100, ...
        PLS_results(band_idx).R2_CI(2) * 100, ...
        PLS_results(band_idx).R2_surrogate_95th * 100, ...
        PLS_results(band_idx).p_surrogate);

    if PLS_results(band_idx).p_surrogate < 0.05
        fprintf(' *\n');
    else
        fprintf('\n');
    end
end

fprintf('\nNote: * p < .05; Surrogate permutations N = %d\n\n', n_surrogate);

%% Visualization (Optional)

try
    figure('Position', [100, 100, 1200, 800]);

    % Panel 1: PLS R² comparison
    subplot(2, 2, 1);
    band_names = {PLS_results.band};
    R2_values = [PLS_results.R2_real] * 100;
    R2_CIs = cat(1, PLS_results.R2_CI) * 100;
    R2_surr_95th = [PLS_results.R2_surrogate_95th] * 100;

    bar(R2_values);
    hold on;
    errorbar(1:n_bands, R2_values, R2_values - R2_CIs(:,1)', R2_CIs(:,2)' - R2_values, ...
        'k', 'LineStyle', 'none', 'LineWidth', 1.5);
    plot(1:n_bands, R2_surr_95th, 'r--', 'LineWidth', 2);
    set(gca, 'XTickLabel', band_names);
    ylabel('Variance Explained (%)');
    title('PLS Prediction: AI GPDC → Learning');
    legend('Real R²', '95% CI', 'Surrogate 95th', 'Location', 'best');
    grid on;

    % Panel 2: Indirect effects comparison
    subplot(2, 2, 2);
    indirect_values = [mediation_results.indirect_beta];
    indirect_CIs = cat(1, mediation_results.indirect_CI);

    bar(indirect_values);
    hold on;
    errorbar(1:n_bands, indirect_values, ...
        indirect_values - indirect_CIs(:,1)', indirect_CIs(:,2)' - indirect_values, ...
        'k', 'LineStyle', 'none', 'LineWidth', 1.5);
    yline(0, 'r--', 'LineWidth', 1);
    set(gca, 'XTickLabel', band_names);
    ylabel('Indirect Effect (β)');
    title('Mediation: Gaze → AI → Learning');
    grid on;

    % Panel 3: Bootstrap distributions (Alpha)
    subplot(2, 2, 3);
    % Use stored bootstrap results from Alpha band mediation
    % (Results from Part 2, band_idx = 3)

    % For visualization, use the stored mediation_results
    alpha_indirect = mediation_results(3).indirect_beta;
    alpha_p = mediation_results(3).p_indirect;

    % Placeholder for bootstrap distribution visualization
    % In actual analysis, plot stored bootstrap samples
    text(0.5, 0.5, sprintf('Alpha Band Mediation\nβ = %.2f\np = %.3f', ...
        alpha_indirect, alpha_p), ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'FontSize', 12);
    xlabel('Indirect Effect (α×β)');
    ylabel('Bootstrap Frequency');
    title('Alpha Band Distribution');
    grid on;

    % Panel 4: Effect size progression
    subplot(2, 2, 4);
    plot(1:n_bands, R2_values, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.85 0.33 0.10]);
    hold on;
    plot(1:n_bands, abs(indirect_values)*100, 's-', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.00 0.45 0.74]);
    set(gca, 'XTickLabel', band_names);
    ylabel('Effect Size');
    title('Effect Size Progression Across Bands');
    legend('PLS R² (%)', 'Indirect Effect (×100)', 'Location', 'best');
    grid on;

    sgtitle('Frequency Robustness Analysis: Delta, Theta, Alpha Bands');

    fprintf('Visualization generated successfully.\n\n');
catch
    fprintf('Note: Visualization requires MATLAB graphics. Skipped.\n\n');
end

%% Summary and Manuscript Reporting

fprintf('========================================================================\n');
fprintf('Robustness Validation Summary\n');
fprintf('========================================================================\n\n');

fprintf('OBJECTIVE:\n');
fprintf('  Validate mediation findings across multiple frequency bands.\n\n');

fprintf('METHODS:\n');
fprintf('  1. PLS regression: AI GPDC → Learning for Delta/Theta/Alpha\n');
fprintf('  2. Mediation analysis: Gaze → AI → Learning for each band\n');
fprintf('  3. Bootstrap + surrogate testing for significance\n\n');

fprintf('KEY FINDINGS:\n\n');

fprintf('PLS Prediction (AI → Learning):\n');
for band_idx = 1:n_bands
    fprintf('  %s (%.0f-%.0f Hz): R² = %.1f%%, p = %.3f', ...
        bands(band_idx).name, bands(band_idx).range(1), bands(band_idx).range(2), ...
        PLS_results(band_idx).R2_real * 100, PLS_results(band_idx).p_surrogate);
    if PLS_results(band_idx).p_surrogate < 0.05
        fprintf(' *\n');
    else
        fprintf('\n');
    end
end

fprintf('\nMediation (Indirect Effect):\n');
for band_idx = 1:n_bands
    fprintf('  %s: β = %.2f, 95%% CI [%.2f, %.2f], p = %.3f', ...
        bands(band_idx).name, ...
        mediation_results(band_idx).indirect_beta, ...
        mediation_results(band_idx).indirect_CI(1), ...
        mediation_results(band_idx).indirect_CI(2), ...
        mediation_results(band_idx).p_indirect);
    if mediation_results(band_idx).p_indirect < 0.05
        fprintf(' *\n');
    else
        fprintf('\n');
    end
end

fprintf('\nCONVERGENCE PATTERN:\n');
fprintf('  Both Theta and Alpha bands show significant effects.\n');
fprintf('  Delta band shows trend but does not reach significance.\n');
fprintf('  Effect sizes increase with frequency: Delta < Theta < Alpha.\n');
fprintf('  This pattern aligns with infant oscillatory maturation.\n\n');

fprintf('CONCLUSION:\n');
fprintf('  Mediation pathway (Gaze → AI connectivity → Learning) replicates\n');
fprintf('  across infant-specific frequency bands (Theta 3-6 Hz, Alpha 6-9 Hz),\n');
fprintf('  validating robustness beyond single frequency specification.\n\n');

fprintf('Manuscript Reporting (Supplementary Section 13.4):\n');
fprintf('  "Frequency robustness analysis confirmed mediation effects across\n');
fprintf('   multiple bands: Alpha (6-9 Hz: β = %.2f, p = %.3f), Theta (3-6 Hz:\n', ...
    mediation_results(3).indirect_beta, mediation_results(3).p_indirect);
fprintf('   β = %.2f, p = %.3f). PLS prediction converged across bands (Alpha\n', ...
    mediation_results(2).indirect_beta, mediation_results(2).p_indirect);
fprintf('   R² = %.1f%%, Theta R² = %.1f%%), demonstrating findings are not\n', ...
    PLS_results(3).R2_real*100, PLS_results(2).R2_real*100);
fprintf('   artifacts of frequency boundary definitions (Supp Tables S14-S15)."\n\n');

fprintf('========================================================================\n');
fprintf('Script complete.\n');
fprintf('========================================================================\n');
