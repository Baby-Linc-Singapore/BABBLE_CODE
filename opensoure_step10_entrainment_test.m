%% Neural Entrainment Statistical Analysis
% Purpose: Statistical analysis of entrainment between speech envelopes and neural oscillations
% in infant EEG data across different experimental conditions (gaze manipulation)
%
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not
% available due to data privacy regulations. Access to anonymized data collected can be requested
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and
% is subject to the establishment of a specific data sharing agreement between the applicant's
% institution and the institutions of data collection.
%
% This script processes pre-computed entrainment measures to:
% 1. Analyze statistical significance using permutation testing
% 2. Apply FDR correction for multiple comparisons
% 3. Visualize entrainment effects across brain regions and frequency bands

%% Load and prepare entrainment data
% Set paths
baseDir = '/path/to/data/';

% Load entrainment data from first phrase only
phrase = 1;
[numData, textData] = xlsread([baseDir, 'analysis/ENTRAINTABLE.xlsx']);
% Select only data from specified phrase
numData = numData(numData(:,3) == phrase,:);

% Separate data by experimental condition
cond1Indices = find(numData(:,2) == 1);
cond2Indices = find(numData(:,2) == 2);
cond3Indices = find(numData(:,2) == 3);

% Load surrogate data from permutation testing
load('ENTRIANSURR.mat', 'permutable')

% Initialize matrices for surrogate data means by condition
surrogateMean1 = [];
surrogateMean2 = [];
surrogateMean3 = [];

% Process surrogate data for each permutation
for i = 1:length(permutable)
    if ~isempty(permutable{i})
        % Extract numerical data from table
        tmpData = permutable{i};
        tmpData = table2array(tmpData(:,2:end));
        
        % Take absolute value of small correlations
        tmpData(abs(tmpData) < 1) = abs(tmpData(abs(tmpData) < 1));
        
        % Select data from specified phrase
        tmpData = tmpData(tmpData(:,3) == phrase,:);
        
        % Group by condition
        tmpCond1 = tmpData(tmpData(:,2) == 1,:);
        tmpCond2 = tmpData(tmpData(:,2) == 2,:);
        tmpCond3 = tmpData(tmpData(:,2) == 3,:);
        
        % Calculate mean for each condition and add to collection
        surrogateMean1 = [surrogateMean1; nanmean(tmpCond1)];
        surrogateMean2 = [surrogateMean2; nanmean(tmpCond2)];
        surrogateMean3 = [surrogateMean3; nanmean(tmpCond3)];
    end
end

%% Define column indices for peak and lag measures
% Column indices for entrainment measures by region and frequency band
peakColumns = [4:12, 22:30, 40:48];  % Correlation peak values
lagColumns = [13:21, 31:39, 49:57];  % Time lag values

%% Process real entrainment data
% Take absolute value of small correlations
numData(abs(numData) < 1) = abs(numData(abs(numData) < 1));

% Calculate means for each condition
realMean1 = nanmean(numData(numData(:,2) == 1,:));
realMean2 = nanmean(numData(numData(:,2) == 2,:));
realMean3 = nanmean(numData(numData(:,2) == 3,:));

%% Statistical testing for Condition 1 - peak values
sigPeakCols1 = [];
peakPValues1 = [];

for i = 1:length(peakColumns)
    col = peakColumns(i);
    % Calculate p-value (proportion of surrogate values >= real value)
    pValue = sum(surrogateMean1(:,col) >= realMean1(col)) / size(surrogateMean1, 1);
    if pValue <= 1
        sigPeakCols1 = [sigPeakCols1, col];
        peakPValues1 = [peakPValues1, pValue];
    end
end

%% Statistical testing for Condition 1 - lag values
sigLagCols1 = [];
lagOutliers1 = [];
lagPValues1 = [];

for i = 1:length(lagColumns)
    col = lagColumns(i);
    % Calculate percentile thresholds for two-tailed test
    lowerPercentile = prctile(surrogateMean1(:,col), 2.5);
    upperPercentile = prctile(surrogateMean1(:,col), 97.5);
    
    if realMean1(col) < lowerPercentile
        pValue = sum(surrogateMean1(:,col) <= realMean1(col)) / size(surrogateMean1, 1);
        outlierStatus = "< 2.5%";
    elseif realMean1(col) > upperPercentile
        pValue = sum(surrogateMean1(:,col) >= realMean1(col)) / size(surrogateMean1, 1);
        outlierStatus = "> 97.5%";
    else
        pValue = 2 * min(sum(surrogateMean1(:,col) <= realMean1(col)), ...
            sum(surrogateMean1(:,col) >= realMean1(col))) / size(surrogateMean1, 1);
        outlierStatus = "Not significant";
    end
    
    if pValue <= 1
        sigLagCols1 = [sigLagCols1, col];
        lagOutliers1 = [lagOutliers1, outlierStatus];
        lagPValues1 = [lagPValues1, pValue];
    end
end

% Apply FDR correction for multiple comparisons
qPeak1 = mafdr(peakPValues1, 'BHFDR', true);
qLag1 = mafdr(lagPValues1, 'BHFDR', true);

%% Statistical testing for Condition 2 - peak values
sigPeakCols2 = [];
peakPValues2 = [];

for i = 1:length(peakColumns)
    col = peakColumns(i);
    pValue = sum(surrogateMean2(:,col) >= realMean2(col)) / size(surrogateMean2, 1);
    if pValue <= 1
        sigPeakCols2 = [sigPeakCols2, col];
        peakPValues2 = [peakPValues2, pValue];
    end
end

%% Statistical testing for Condition 2 - lag values
sigLagCols2 = [];
lagOutliers2 = [];
lagPValues2 = [];

for i = 1:length(lagColumns)
    col = lagColumns(i);
    lowerPercentile = prctile(surrogateMean2(:,col), 2.5);
    upperPercentile = prctile(surrogateMean2(:,col), 97.5);
    
    if realMean2(col) < lowerPercentile
        pValue = sum(surrogateMean2(:,col) <= realMean2(col)) / size(surrogateMean2, 1);
        outlierStatus = "< 2.5%";
    elseif realMean2(col) > upperPercentile
        pValue = sum(surrogateMean2(:,col) >= realMean2(col)) / size(surrogateMean2, 1);
        outlierStatus = "> 97.5%";
    else
        pValue = 2 * min(sum(surrogateMean2(:,col) <= realMean2(col)), ...
            sum(surrogateMean2(:,col) >= realMean2(col))) / size(surrogateMean2, 1);
        outlierStatus = "Not significant";
    end
    
    if pValue <= 1
        sigLagCols2 = [sigLagCols2, col];
        lagOutliers2 = [lagOutliers2, outlierStatus];
        lagPValues2 = [lagPValues2, pValue];
    end
end

% Apply FDR correction
qPeak2 = mafdr(peakPValues2, 'BHFDR', true);
qLag2 = mafdr(lagPValues2, 'BHFDR', true);

%% Statistical testing for Condition 3 - peak values
sigPeakCols3 = [];
peakPValues3 = [];

for i = 1:length(peakColumns)
    col = peakColumns(i);
    pValue = sum(surrogateMean3(:,col) >= realMean3(col)) / size(surrogateMean3, 1);
    if pValue <= 1
        sigPeakCols3 = [sigPeakCols3, col];
        peakPValues3 = [peakPValues3, pValue];
    end
end

%% Statistical testing for Condition 3 - lag values
sigLagCols3 = [];
lagOutliers3 = [];
lagPValues3 = [];

for i = 1:length(lagColumns)
    col = lagColumns(i);
    lowerPercentile = prctile(surrogateMean3(:,col), 2.5);
    upperPercentile = prctile(surrogateMean3(:,col), 97.5);
    
    if realMean3(col) < lowerPercentile
        pValue = sum(surrogateMean3(:,col) <= realMean3(col)) / size(surrogateMean3, 1);
        outlierStatus = "< 2.5%";
    elseif realMean3(col) > upperPercentile
        pValue = sum(surrogateMean3(:,col) >= realMean3(col)) / size(surrogateMean3, 1);
        outlierStatus = "> 97.5%";
    else
        pValue = 2 * min(sum(surrogateMean3(:,col) <= realMean3(col)), ...
            sum(surrogateMean3(:,col) >= realMean3(col))) / size(surrogateMean3, 1);
        outlierStatus = "Not significant";
    end
    
    if pValue <= 1
        sigLagCols3 = [sigLagCols3, col];
        lagOutliers3 = [lagOutliers3, outlierStatus];
        lagPValues3 = [lagPValues3, pValue];
    end
end

% Apply FDR correction
qPeak3 = mafdr(peakPValues3, 'BHFDR', true);
qLag3 = mafdr(lagPValues3, 'BHFDR', true);

%% Display significant results
% Display results for Condition 1
fprintf('Results for Condition 1:\n');
fprintf('Significant peak columns after FDR correction (q <= 0.05):\n');
for i = 1:length(sigPeakCols1)
    col = sigPeakCols1(i);
    if qPeak1(i) <= 0.05
        fprintf('%s (Column %d): q = %.4f\n', textData{1, col + 1}, col, qPeak1(i));
    end
end
fprintf('\nSignificant lag columns after FDR correction (q <= 0.05):\n');
for i = 1:length(sigLagCols1)
    col = sigLagCols1(i);
    if qLag1(i) <= 0.05
        fprintf('%s (Column %d): %s, q = %.4f\n', textData{1, col + 1}, col, lagOutliers1{i}, qLag1(i));
    end
end
fprintf('\n');

% Display results for Condition 2
fprintf('Results for Condition 2:\n');
fprintf('Significant peak columns after FDR correction (q <= 0.05):\n');
for i = 1:length(sigPeakCols2)
    col = sigPeakCols2(i);
    if qPeak2(i) <= 0.05
        fprintf('%s (Column %d): q = %.4f\n', textData{1, col + 1}, col, qPeak2(i));
    end
end
fprintf('\nSignificant lag columns after FDR correction (q <= 0.05):\n');
for i = 1:length(sigLagCols2)
    col = sigLagCols2(i);
    if qLag2(i) <= 0.05
        fprintf('%s (Column %d): %s, q = %.4f\n', textData{1, col + 1}, col, lagOutliers2{i}, qLag2(i));
    end
end
fprintf('\n');

% Display results for Condition 3
fprintf('Results for Condition 3:\n');
fprintf('Significant peak columns after FDR correction (q <= 0.05):\n');
for i = 1:length(sigPeakCols3)
    col = sigPeakCols3(i);
    if qPeak3(i) <= 0.05
        fprintf('%s (Column %d): q = %.4f\n', textData{1, col + 1}, col, qPeak3(i));
    end
end
fprintf('\nSignificant lag columns after FDR correction (q <= 0.05):\n');
for i = 1:length(sigLagCols3)
    col = sigLagCols3(i);
    if qLag3(i) <= 0.05
        fprintf('%s (Column %d): %s, q = %.4f\n', textData{1, col + 1}, col, lagOutliers3{i}, qLag3(i));
    end
end
fprintf('\n');

%% Create visualization for statistical results
% Create heatmap visualization of effect sizes across channels and frequency bands
figure('Position', [100, 100, 1600, 500]);
createStatisticalMatrices(peakPValues1, qPeak1, realMean1, surrogateMean1, peakColumns, ...
                          peakPValues2, qPeak2, realMean2, surrogateMean2, ...
                          peakPValues3, qPeak3, realMean3, surrogateMean3);

%% Helper Functions

% Function to calculate Cohen's d as effect size metric
function cohensD = calculateCohensD(realMean, surrogateMean)
    [nSamples, nCols] = size(surrogateMean);
    cohensD = zeros(1, nCols);
    
    for i = 1:nCols
        group1 = repmat(realMean(i), nSamples, 1);
        group2 = surrogateMean(:,i);
        
        % Use standard deviation of surrogate data only
        s2 = std(group2);
        
        % Calculate effect size
        d = (mean(group1) - mean(group2)) / s2;
        
        cohensD(i) = d;
    end
end

% Function to plot a single matrix of results
function h = plotMatrix(subplotIdx, pvalues, qvalues, realMean, surrogateMean, cols)
    h = subplot(1, 3, subplotIdx);
    
    % Calculate Cohen's d effect sizes
    cohensD = calculateCohensD(realMean(cols), surrogateMean(:,cols));
    
    % Reshape data for 9x3 matrix (channels x frequency bands)
    cohensD_matrix = reshape(cohensD, [9, 3]);
    qvalues_matrix = reshape(qvalues, [9, 3]);
    
    % Reverse column order for display (Delta, Theta, Alpha)
    cohensD_matrix = fliplr(cohensD_matrix);
    qvalues_matrix = fliplr(qvalues_matrix);
    
    % Display the heatmap
    imagesc(cohensD_matrix);
    
    % Add statistical significance markers
    [rows, cols] = size(cohensD_matrix);
    hold on;
    for i = 1:rows
        for j = 1:cols
            % Add significance stars based on q-value
            if qvalues_matrix(i,j) < 0.05
                % text(j, i, '*', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'k', 'FontSize', 50, 'FontWeight', 'bold');
            end
        end
    end
    hold off;
    
    % Customize plot appearance
    set(gca, 'XTick', 1:cols, 'YTick', 1:rows);
    set(gca, 'XTickLabel', {'Delta', 'Theta', 'Alpha'});
    set(gca, 'YTickLabel', {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'});
    
    if subplotIdx == 1
        ylabel('Channel', 'FontSize', 16, 'FontWeight', 'bold');
    end
    if subplotIdx == 2
        xlabel('Frequency Band', 'FontSize', 20, 'FontWeight', 'bold');
    end
    
    % Enhance visualization
    set(gca, 'FontSize', 20, 'FontWeight', 'bold');
    set(gca, 'TickDir', 'out');
    set(gca, 'LineWidth', 3);
    set(gca, 'LooseInset', get(gca, 'TightInset'));
end

% Function to create statistical matrices for all conditions
function createStatisticalMatrices(peak_p1, q_peak1, realMean1, surrogateMean1, peakColumns, ...
                                  peak_p2, q_peak2, realMean2, surrogateMean2, ...
                                  peak_p3, q_peak3, realMean3, surrogateMean3)
    % Plot statistical matrices for each condition
    h1 = plotMatrix(1, peak_p1, q_peak1, realMean1, surrogateMean1, peakColumns);
    h2 = plotMatrix(2, peak_p2, q_peak2, realMean2, surrogateMean2, peakColumns);
    h3 = plotMatrix(3, peak_p3, q_peak3, realMean3, surrogateMean3, peakColumns);
    
    % Create colormap and colorbar
    colormap(redblue(256));
    c = colorbar;
    
    % Set unified color scale across subplots
    maxAbsD = max([max(abs(get(h1, 'CLim'))), max(abs(get(h2, 'CLim'))), max(abs(get(h3, 'CLim')))]);
    set([h1 h2 h3], 'CLim', [-maxAbsD, maxAbsD]);
    
    % Customize colorbar
    c.LineWidth = 3;
    c.FontSize = 20;
    c.FontWeight = 'bold';
    c.Label.String = 'Cohen''s d';
    c.Label.FontSize = 20;
    c.Label.FontWeight = 'bold';
    
    % Position the colorbar
    cPos = c.Position;
    c.Position = [cPos(1)+0.08 cPos(2) cPos(3)*1 cPos(4)];
end

% Custom colormap function for visualization
function cmap = redblue(m)
    if nargin < 1, m = 64; end
    
    % Create gradient from white to red
    r = ones(m, 1);
    g = linspace(1, 0.2, m)';
    b = g;
    
    cmap = [r g b];
end