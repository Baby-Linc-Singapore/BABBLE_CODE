%% Neural Connectivity and Learning Outcomes Analysis
% Purpose: Analyze the relationship between different types of neural connectivity 
% (adult-infant and infant-infant) and learning outcomes in infants using PLS regression
%
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not
% available due to data privacy regulations. Access to anonymized data collected can be requested
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and
% is subject to the establishment of a specific data sharing agreement between the applicant's
% institution and the institutions of data collection.
%
% This script analyzes:
% 1. How adult-infant (AI) and infant-infant (II) GPDC connectivity predict learning outcomes
% 2. How neural connectivity patterns predict infant CDI gesture scores
% 3. Cross-validation of these predictive relationships

%% Load data and prepare variables
clc
clear all

% Load entrainment and GPDC data
load('ENTRIANSURR.mat');  % Surrogate data from permutation testing
load('dataGPDC.mat');     % GPDC connectivity data
load('CDI2.mat');         % Child Development Inventory data

% Setup channel indices for different connectivity types and frequency bands
ii_alpha = [10+81*2:9+81*3];  % Infant-infant alpha band connectivity
ai_alpha = [10+81*8:9+81*9];  % Adult-infant alpha band connectivity

% Load significant connections identified through FDR correction
listi = ii_alpha;
load('stronglistfdr5_gpdc_II.mat');  % Load significant II connections
listii = listi(s4);
ii = sqrt(data(:,listii));  % Extract and transform II connectivity values

listi = ai_alpha;
load('stronglistfdr5_gpdc_AI.mat');  % Load significant AI connections
listai = listi(s4);
ai = sqrt(data(:,listai));  % Extract and transform AI connectivity values

% Prepare surrogate data
ai_surr = cell(1000,1);
ii_surr = cell(1000,1);
for i = 1:1000
    tmp = data_surr{i};
    ai_surr{i} = sqrt(tmp(:,listai));
    ii_surr{i} = sqrt(tmp(:,listii));
end

%% Figure 4A: PLS analysis of AI GPDC connectivity predicting learning
colorlist = {[252/255, 140/255, 90/255], [226/255, 90/255, 80/255], ...
             [75/255, 116/255, 178/255], [144/255, 190/255, 224/255]};

% PLS analysis with different numbers of components
compall = 10;  % Maximum number of components to test
plotk = zeros(compall, 4);
valid = find(~isnan(learning));

% Calculate variance explained by real data for each number of components
for comp = 1:compall  
    [~, ~, ~, ~, ~, PCTVAR, ~, ~] = plsregress(zscore([ai(valid,:), data(valid, [1, 3, 4])]), ...
                                              zscore(learning(valid)), comp);
    PCTVAR_real2 = sum(PCTVAR(2, :));
    plotk(comp, 1) = PCTVAR_real2;

    % Calculate variance explained by surrogate data
    PCTVAR_su2 = zeros(1000, 1);
    for i = 1:1000
        ais = ai_surr{i};
        [~, ~, ~, ~, ~, PCTVAR, ~, ~] = plsregress(zscore([ais(valid,:), data(valid, [1, 3, 4])]), ...
                                                  zscore(learning(valid)), comp);
        PCTVAR_su2(i) = sum(PCTVAR(2, :));
    end

    plotk(comp, 2) = prctile(PCTVAR_su2, 95);  % 95th percentile of surrogates
    plotk(comp, 3) = mean(PCTVAR_su2);         % Mean of surrogates
    plotk(comp, 4) = prctile(PCTVAR_su2, 0);   % Minimum of surrogates
end

% Create Figure 4A: AI GPDC predicting learning
figure('Position', [100, 100, 800, 600]);
hold on;

% Plot shaded area representing surrogate distribution
fill([1:compall, fliplr(1:compall)], [plotk(:, 2)', fliplr(plotk(:, 4)')], ...
     [200/255, 200/255, 200/255], 'EdgeColor', 'none');

% Plot mean of surrogates
plot(1:compall, plotk(:, 3), 'k', 'LineWidth', 3);

% Plot real data
plot(1:compall, plotk(:, 1), 'Color', [75/255, 116/255, 178/255], 'LineWidth', 3);

% Format plot
ax = gca;
ax.Box = 'on';
ax.LineWidth = 2;
ax.FontName = 'Arial';
ax.FontSize = 16;
ax.FontWeight = 'bold';
ax.TickDir = 'in';
ax.Layer = 'top';

xlabel('Component number', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Learning variance explained', 'FontSize', 20, 'FontWeight', 'bold');
xlim([1, compall]);

% Convert y-axis to percentages
yticks = ax.YTick;
ax.YTickLabel = strcat(string(yticks * 100), '%');

legend({'0-95% CI of surrogates', 'Mean of surrogates', 'Real AI GPDC'}, ...
       'Location', 'southeast', 'FontSize', 16);
title('Learning prediction performance by AI GPDC', 'FontSize', 20, 'FontWeight', 'bold');
hold off;

%% Figure 4B: PLS analysis of II GPDC connectivity predicting CDI gesture scores
compall = 10;  % Maximum number of components to test
plotk = zeros(compall, 4);
valid = find(~isnan(CDIG));

% Calculate variance explained by real data for each number of components
for comp = 1:compall  
    [~, ~, ~, ~, ~, PCTVAR, ~, ~] = plsregress(zscore([ii(valid,:), data(valid, [1, 3, 4])]), ...
                                              zscore(CDIG(valid)), comp);
    PCTVAR_real2 = sum(PCTVAR(2, :));
    plotk(comp, 1) = PCTVAR_real2;

    % Calculate variance explained by surrogate data
    PCTVAR_su2 = zeros(1000, 1);
    for i = 1:1000
        iis = ii_surr{i};
        [~, ~, ~, ~, ~, PCTVAR, ~, ~] = plsregress(zscore([iis(valid,:), data(valid, [1, 3, 4])]), ...
                                                  zscore(CDIG(valid)), comp);
        PCTVAR_su2(i) = sum(PCTVAR(2, :));
    end

    plotk(comp, 2) = prctile(PCTVAR_su2, 95);  % 95th percentile of surrogates
    plotk(comp, 3) = mean(PCTVAR_su2);         % Mean of surrogates
    plotk(comp, 4) = prctile(PCTVAR_su2, 0);   % Minimum of surrogates
end

% Create Figure 4B: II GPDC predicting CDI gesture scores
figure('Position', [100, 100, 800, 600]);
hold on;

% Plot shaded area representing surrogate distribution
fill([1:compall, fliplr(1:compall)], [plotk(:, 2)', fliplr(plotk(:, 4)')], ...
     [200/255, 200/255, 200/255], 'EdgeColor', 'none');

% Plot mean of surrogates
plot(1:compall, plotk(:, 3), 'k', 'LineWidth', 3);

% Plot real data
plot(1:compall, plotk(:, 1), 'Color', [252/255, 140/255, 90/255], 'LineWidth', 3);

% Format plot
ax = gca;
ax.Box = 'on';
ax.LineWidth = 2;
ax.FontName = 'Arial';
ax.FontSize = 16;
ax.FontWeight = 'bold';
ax.TickDir = 'in';
ax.Layer = 'top';

xlabel('Component number', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('CDI-G variance explained', 'FontSize', 20, 'FontWeight', 'bold');
xlim([1, compall]);

% Convert y-axis to percentages
yticks = ax.YTick;
ax.YTickLabel = strcat(string(yticks * 100), '%');

legend({'0-95% CI of surrogates', 'Mean of surrogates', 'Real II GPDC'}, ...
       'Location', 'southeast', 'FontSize', 16);
title('CDI-G prediction performance by II GPDC', 'FontSize', 20, 'FontWeight', 'bold');
hold off;

%% Cross-validation of PLS model performance
% Bootstrap cross-validation of AI GPDC predicting CDI gesture scores
n_folds = 10;
n_bootstrap = 1000;
mean_variance_explained_bootstrap = zeros(n_bootstrap, 1);

% Prepare data for bootstrapping
SEX = double(SEX);
COUNTRY = double(COUNTRY);
tbl = table(ID, atten, zscore(learning), zscore(AGE), SEX, COUNTRY, blocks, CONDGROUP, ai, ...
    'VariableNames', {'ID','atten', 'learning', 'AGE', 'SEX', 'COUNTRY', 'block', 'CONDGROUP', 'ai'});

% Add CDI data to table
for i = 1:size(tbl,1)
    id = a(i,2);
    for j = 1:47
        if a2(j,1) == id
            tbl.CDIP(i) = a2(j,2);
            tbl.CDIW(i) = a2(j,3);
            tbl.CDIG(i) = a2(j,4);
        end
    end
end

% Perform bootstrap cross-validation
for boot = 1:n_bootstrap
    % Create bootstrap sample
    bootstrap_idx = datasample(1:height(tbl), height(tbl), 'Replace', true);
    tbl_bootstrap = tbl(bootstrap_idx, :);
    
    % Remove rows with missing CDIG values
    valid = find(~isnan(tbl_bootstrap.CDIG));
    tbl_bootstrap = tbl_bootstrap(valid,:);
    
    % Initialize storage for variance explained in each fold
    variance_explained = zeros(n_folds, 1);
    
    % Create cross-validation partition
    cv = cvpartition(height(tbl_bootstrap), 'KFold', n_folds);
    
    for fold = 1:n_folds
        % Get training and test indices
        train_idx = training(cv, fold);
        test_idx = test(cv, fold);
        
        % Prepare training data
        X_train = [tbl_bootstrap.ai(train_idx,:), tbl_bootstrap.AGE(train_idx), ...
                  tbl_bootstrap.SEX(train_idx), tbl_bootstrap.COUNTRY(train_idx)];
        Y_train = tbl_bootstrap.CDIG(train_idx);
        
        % Train PLS model
        n_components = 1;
        [~, ~, ~, ~, ~, ~, ~, stats] = plsregress(X_train, Y_train, n_components);
        
        % Prepare test data
        X_test = [tbl_bootstrap.ai(test_idx,:), tbl_bootstrap.AGE(test_idx), ...
                 tbl_bootstrap.SEX(test_idx), tbl_bootstrap.COUNTRY(test_idx)];
        Y_test = tbl_bootstrap.CDIG(test_idx);
        
        % Calculate test scores
        scores_test = X_test * stats.W;
        
        % Calculate variance explained
        variance_explained(fold) = corr(Y_test, scores_test(:, 1))^2;
    end
    
    % Store mean variance explained for this bootstrap
    mean_variance_explained_bootstrap(boot) = mean(variance_explained);
    fprintf('Bootstrap iteration %d complete\n', boot);
end

% Calculate overall results
mean_variance_explained = mean(mean_variance_explained_bootstrap);
std_variance_explained = std(mean_variance_explained_bootstrap);

fprintf('\nBootstrap Results (1000 iterations) of Mean variance explained with 10-Fold CV:\n');
fprintf('Mean of Mean variance explained = %f, Std of Mean variance explained = %f\n', ...
        mean_variance_explained, std_variance_explained);
fprintf('Range of Mean variance explained = [%f, %f]\n', ...
        min(mean_variance_explained_bootstrap), max(mean_variance_explained_bootstrap));

%% Figure 4C: Visualization of component loadings
% Bootstrap to obtain stable component loadings
n_nodes = 64;
n_components = 1;
n_iterations = 1000;
valid = find(~isnan(CDIG));

% Initialize storage for bootstrap weights
bootstrap_weights = zeros(n_nodes, n_components, n_iterations);
X_train = zscore([ii(valid,:), data(valid, [1, 3, 4])]);
Y_train = zscore(CDIG(valid));

for iter = 1:n_iterations
    % Perform bootstrap sampling
    sample_idx = randi(size(X_train, 1), size(X_train, 1), 1);
    X_bootstrap = X_train(sample_idx, :);
    Y_bootstrap = Y_train(sample_idx);
    
    % Fit PLS model
    [XL, ~, ~, ~, ~, ~, ~, ~] = plsregress(X_bootstrap, Y_bootstrap, n_components);
    
    % Store weights
    bootstrap_weights(:, :, iter) = XL(1:n_nodes,:);
end

% Calculate mean and standard deviation of weights
mean_bootstrap_weights = mean(bootstrap_weights, 3);
std_bootstrap_weights = std(bootstrap_weights, [], 3);

% Calculate standardized weights (z-scores)
z_scores = mean_bootstrap_weights ./ std_bootstrap_weights;

% Get absolute values for visualization
loadings = abs(z_scores);

% Create matrix representation for the connectivity pattern
labels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
connectivity_matrix = zeros(9, 9);

% Fill the matrix with loadings values from significant connections
full_matrix_ii = zeros(81, 1);
full_matrix_ii(s4) = 1;
connectivity_matrix(full_matrix_ii == 1) = loadings;

% Visualize the connectivity matrix
figure;
imagesc(connectivity_matrix);

% Format plot
set(gca, 'XTick', 1:9, 'XTickLabel', labels, 'FontWeight', 'bold', 'FontName', 'Arial', 'FontSize', 14);
set(gca, 'YTick', 1:9, 'YTickLabel', labels, 'FontWeight', 'bold', 'FontName', 'Arial', 'FontSize', 14);

xlabel('Sender channels', 'FontWeight', 'bold', 'FontSize', 20, 'FontName', 'Arial');
ylabel('Receiver channels', 'FontWeight', 'bold', 'FontSize', 20, 'FontName', 'Arial');
title('Component No.1 absolute loadings for II GPDC', 'FontWeight', 'bold', 'FontSize', 20, 'FontName', 'Arial');

% Custom colormap (white to red)
num_colors = 256;
white = [1 1 1];
pink = [1 0.2 0.2];
custom_colormap = [linspace(white(1), pink(1), num_colors)', ...
                  linspace(white(2), pink(2), num_colors)', ...
                  linspace(white(3), pink(3), num_colors)'];
colormap(custom_colormap);

% Add colorbar
h = colorbar;
set(h, 'FontSize', 14, 'FontName', 'Arial', 'FontWeight', 'bold', 'LineWidth', 2);
set(gca, 'LineWidth', 2);