%% Mediation Analysis of Gaze, Neural Connectivity, and Learning
% Purpose: Examine how gaze conditions affect infant learning through neural mechanisms
%          using mediation analysis with bootstrapping
%
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not
% available due to data privacy regulations. Access to anonymized data collected can be requested
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and
% is subject to the establishment of a specific data sharing agreement between the applicant's
% institution and the institutions of data collection.
%
% This script performs mediation analysis to:
% 1. Test if neural synchrony measures mediate the relationship between gaze condition and learning
% 2. Test if adult-infant connectivity mediates the relationship between gaze condition and learning
% 3. Compare the direct and indirect effects through bootstrap confidence intervals

%% Load and prepare data
clc
clear all

% Load connectivity and entrainment data
load('ENTRIANSURR.mat');
load('dataGPDC.mat');

% Prepare demographics and outcome variables
AGE = data(:,3);                    % Age in months
SEX = categorical(data(:,4));        % Sex (categorical)
COUNTRY = categorical(data(:,1));    % Data collection site
learning = data(:,7);               % Learning outcome measure
atten = data(:,9);                  % Attention measure
ID = categorical(data(:,2));         % Participant ID

% Define experimental conditions
COND = categorical(data(:,6));      % Experimental gaze condition

% Define indices for connectivity data by frequency and connection type
% II = infant-infant, AA = adult-adult, AI = adult-infant, IA = infant-adult
ii_alpha = [10+81*2:9+81*3];  % Infant-infant alpha band indices
ai_alpha = [10+81*8:9+81*9];  % Adult-infant alpha band indices
aa_alpha = [10+81*5:9+81*6];  % Adult-adult alpha band indices

% Load significant connections identified through FDR correction
load('stronglistfdr5_gpdc_II.mat');
listii = ii_alpha(s4);
ii = sqrt(data(:,listii));  % Extract II alpha connectivity

load('stronglistfdr5_gpdc_AI.mat');
listai = ai_alpha(s4);
ai = sqrt(data(:,listai));  % Extract AI alpha connectivity

load('stronglistfdr5_gpdc_AA.mat');
listaa = aa_alpha(s4);
aa = sqrt(data(:,listaa));  % Extract AA alpha connectivity

% Load neural entrainment data
[num_data, txt_data] = xlsread('TABLE.xlsx');
entrain_features = num_data(:,[4,5,21,26,40]);  % Selected entrainment features:
% 1. Alpha C3 entrainment
% 2. Alpha Cz entrainment
% 3. Theta F4 entrainment
% 4. Theta Pz entrainment
% 5. Delta C3 entrainment

% Find valid data points (non-NaN)
valid = find(~isnan(entrain_features(:,1)));

%% Prepare data for mediation analysis

% Create PLS component for adult-infant connectivity
[~, ~, XS_ai, ~, ~, ~, ~, ~] = plsregress(zscore([ai, data(:,[1,3,4])]), zscore(learning), 2);
ai_comp = zscore(XS_ai(:,1));  % First PLS component of AI connectivity

% Create PLS component for infant-infant connectivity
[~, ~, XS_ii, ~, ~, ~, ~, ~] = plsregress(zscore([ii, data(:,[1,3,4])]), zscore(learning), 2);
ii_comp = zscore(XS_ii(:,1));  % First PLS component of II connectivity

% Create table for analysis
tbl = table(ID, atten, entrain_features(:,5), zscore(learning), zscore(AGE), SEX, COUNTRY, COND, ai_comp, ii_comp, ...
    'VariableNames', {'ID', 'atten', 'entrain', 'learning', 'AGE', 'SEX', 'COUNTRY', 'COND', 'ai', 'ii'});

% Standardize entrainment measure for valid data points
tbl.entrain(valid) = (tbl.entrain(valid) - min(tbl.entrain(valid))) ./ std(tbl.entrain(valid));

%% Mediation analysis: Gaze → AI connectivity → Learning

% Step 1: Test if AI connectivity predicts learning
m1 = fitlme(tbl, 'learning ~ ai + (1|ID)');

% Step 2: Test if gaze condition predicts AI connectivity, and if entrainment moderates this relationship
m2 = fitlme(tbl, 'ai ~ entrain * COND + (1|ID)');

% Step 3: Test if gaze condition has a direct effect on learning, controlling for AI connectivity
m3 = fitlme(tbl, 'learning ~ ai + COND + (1|ID)');

% Display model results
disp('Mediation Analysis Results:');
disp('Model 1: AI connectivity predicting learning');
disp(m1.Coefficients);
disp('Model 2: Gaze condition predicting AI connectivity with entrainment moderation');
disp(m2.Coefficients);
disp('Model 3: Direct effect of gaze on learning controlling for AI connectivity');
disp(m3.Coefficients);

%% Bootstrap analysis of direct and indirect effects in AI mediation

% Number of bootstrap iterations
numBoot = 1000;

% Pre-allocate storage for coefficients
indirectEffects_ai = zeros(numBoot, 1);  % Stores indirect effect coefficients
directEffects_ai = zeros(numBoot, 1);    % Stores direct effect coefficients

% Bootstrap iterations
for i = 1:numBoot
    % Resample with replacement
    bootIdx = randsample(height(tbl), height(tbl), true);
    tbl_boot = tbl(bootIdx, :);
    
    % Fit the first model: AI connectivity predicting learning
    m1_boot = fitlme(tbl_boot, 'learning ~ ai + (1|ID)');
    ai_coef = m1_boot.Coefficients.Estimate(strcmp(m1_boot.Coefficients.Name, 'ai'));
    
    % Fit the second model: Gaze condition predicting AI connectivity
    m2_boot = fitlme(tbl_boot, 'ai ~ entrain * COND + (1|ID)');
    % Extract coefficient for the gaze condition of interest
    gaze_coef = m2_boot.Coefficients.Estimate(3); % Adjust index based on your model
    
    % Calculate indirect effect and store
    indirectEffects_ai(i) = ai_coef * gaze_coef;
    
    % Fit the third model: Direct effect of gaze on learning
    m3_boot = fitlme(tbl_boot, 'learning ~ ai + COND + (1|ID)');
    % Extract coefficient for the direct effect
    directEffects_ai(i) = m3_boot.Coefficients.Estimate(3); % Adjust index based on your model
end

% Adjust sign of indirect effects if needed
indirectEffects_ai = -indirectEffects_ai;

% Calculate 95% confidence intervals
indirectEffect_ai_CI = prctile(indirectEffects_ai, [2.5, 97.5]);
directEffect_ai_CI = prctile(directEffects_ai, [2.5, 97.5]);

% Display results
fprintf('\nBootstrap Analysis Results for AI Mediation (1000 iterations):\n');
fprintf('Indirect Effect (AI mediation) Mean: %f, SD: %f\n', mean(indirectEffects_ai), std(indirectEffects_ai));
fprintf('Indirect Effect 95%% CI: [%f, %f]\n', indirectEffect_ai_CI(1), indirectEffect_ai_CI(2));
fprintf('Direct Effect Mean: %f, SD: %f\n', mean(directEffects_ai), std(directEffects_ai));
fprintf('Direct Effect 95%% CI: [%f, %f]\n', directEffect_ai_CI(1), directEffect_ai_CI(2));
fprintf('Proportion of negative indirect effects: %f\n', sum(indirectEffects_ai < 0) / numBoot);
fprintf('Proportion of negative direct effects: %f\n', sum(directEffects_ai < 0) / numBoot);

%% Mediation analysis: Gaze → Neural entrainment → Learning

% Step 1: Test if entrainment predicts learning
m1 = fitlme(tbl, 'learning ~ entrain + AGE + SEX + COUNTRY + (1|ID)');

% Step 2: Test if gaze condition predicts entrainment
m2 = fitlme(tbl, 'entrain ~ COND + AGE + SEX + COUNTRY + (1|ID)');

% Step 3: Test if gaze condition has a direct effect on learning, controlling for entrainment
m3 = fitlme(tbl, 'learning ~ entrain + COND + AGE + SEX + COUNTRY + (1|ID)');

% Display model results
disp('Mediation Analysis Results for Entrainment:');
disp('Model 1: Entrainment predicting learning');
disp(m1.Coefficients);
disp('Model 2: Gaze condition predicting entrainment');
disp(m2.Coefficients);
disp('Model 3: Direct effect of gaze on learning controlling for entrainment');
disp(m3.Coefficients);

%% Bootstrap analysis of direct and indirect effects in entrainment mediation

% Pre-allocate storage for coefficients
indirectEffects_nse = zeros(numBoot, 1);  % Stores indirect effect coefficients
directEffects_nse = zeros(numBoot, 1);    % Stores direct effect coefficients

% Bootstrap iterations
for i = 1:numBoot
    % Resample with replacement
    bootIdx = randsample(height(tbl), height(tbl), true);
    tbl_boot = tbl(bootIdx, :);
    
    % Fit the first model: Entrainment predicting learning
    m1_boot = fitlme(tbl_boot, 'learning ~ entrain + AGE + SEX + COUNTRY + (1|ID)');
    entrain_coef = m1_boot.Coefficients.Estimate(strcmp(m1_boot.Coefficients.Name, 'entrain'));
    
    % Fit the second model: Gaze condition predicting entrainment
    m2_boot = fitlme(tbl_boot, 'entrain ~ COND + AGE + SEX + COUNTRY + (1|ID)');
    % Extract coefficient for the gaze condition of interest
    gaze_coef = m2_boot.Coefficients.Estimate(2); % Adjust index based on your model
    
    % Calculate indirect effect and store
    indirectEffects_nse(i) = entrain_coef * gaze_coef;
    
    % Fit the third model: Direct effect of gaze on learning
    m3_boot = fitlme(tbl_boot, 'learning ~ entrain + COND + AGE + SEX + COUNTRY + (1|ID)');
    % Extract coefficient for the direct effect
    directEffects_nse(i) = m3_boot.Coefficients.Estimate(strcmp(m3_boot.Coefficients.Name, 'COND_2'));
end

% Adjust sign of indirect effects if needed
indirectEffects_nse = -indirectEffects_nse;

% Calculate 95% confidence intervals
indirectEffect_nse_CI = prctile(indirectEffects_nse, [2.5, 97.5]);
directEffect_nse_CI = prctile(directEffects_nse, [2.5, 97.5]);

% Display results
fprintf('\nBootstrap Analysis Results for Entrainment Mediation (1000 iterations):\n');
fprintf('Indirect Effect (Entrainment mediation) Mean: %f, SD: %f\n', mean(indirectEffects_nse), std(indirectEffects_nse));
fprintf('Indirect Effect 95%% CI: [%f, %f]\n', indirectEffect_nse_CI(1), indirectEffect_nse_CI(2));
fprintf('Direct Effect Mean: %f, SD: %f\n', mean(directEffects_nse), std(directEffects_nse));
fprintf('Direct Effect 95%% CI: [%f, %f]\n', directEffect_nse_CI(1), directEffect_nse_CI(2));
fprintf('Proportion of negative indirect effects: %f\n', sum(indirectEffects_nse < 0) / numBoot);
fprintf('Proportion of negative direct effects: %f\n', sum(directEffects_nse < 0) / numBoot);

%% Additional analysis: Investigate II connectivity as a mediator (for comparison)

% Mediation analysis steps for II connectivity
m1 = fitlme(tbl, 'learning ~ ii + (1|ID)');
m2 = fitlme(tbl, 'ii ~ entrain * COND + (1|ID)');
m3 = fitlme(tbl, 'learning ~ ii + COND + (1|ID)');

% Bootstrap analysis for II connectivity mediation
indirectEffects_ii = zeros(numBoot, 1);
directEffects_ii = zeros(numBoot, 1);

for i = 1:numBoot
    bootIdx = randsample(height(tbl), height(tbl), true);
    tbl_boot = tbl(bootIdx, :);
    
    m1_boot = fitlme(tbl_boot, 'learning ~ ii + (1|ID)');
    ii_coef = m1_boot.Coefficients.Estimate(strcmp(m1_boot.Coefficients.Name, 'ii'));
    
    m2_boot = fitlme(tbl_boot, 'ii ~ entrain * COND + (1|ID)');
    gaze_coef = m2_boot.Coefficients.Estimate(3);
    
    indirectEffects_ii(i) = ii_coef * gaze_coef;
    
    m3_boot = fitlme(tbl_boot, 'learning ~ ii + COND + (1|ID)');
    directEffects_ii(i) = m3_boot.Coefficients.Estimate(3);
end

indirectEffects_ii = -indirectEffects_ii;

% Calculate 95% confidence intervals
indirectEffect_ii_CI = prctile(indirectEffects_ii, [2.5, 97.5]);
directEffect_ii_CI = prctile(directEffects_ii, [2.5, 97.5]);

fprintf('\nBootstrap Analysis Results for II Connectivity Mediation (1000 iterations):\n');
fprintf('Indirect Effect (II mediation) Mean: %f, SD: %f\n', mean(indirectEffects_ii), std(indirectEffects_ii));
fprintf('Indirect Effect 95%% CI: [%f, %f]\n', indirectEffect_ii_CI(1), indirectEffect_ii_CI(2));
fprintf('Direct Effect Mean: %f, SD: %f\n', mean(directEffects_ii), std(directEffects_ii));
fprintf('Direct Effect 95%% CI: [%f, %f]\n', directEffect_ii_CI(1), directEffect_ii_CI(2));
fprintf('Proportion of negative indirect effects: %f\n', sum(indirectEffects_ii < 0) / numBoot);
fprintf('Proportion of negative direct effects: %f\n', sum(directEffects_ii < 0) / numBoot);
