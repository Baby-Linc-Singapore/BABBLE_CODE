%% GPDC Connectivity Analysis Script
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Calculate Generalized Partial Directed Coherence (GPDC) from EEG data
% to examine neural connectivity patterns between adult and infant dyads
%
% This script:
% 1. Processes adult and infant EEG data from two datasets
% 2. Calculates power spectral densities in different frequency bands
% 3. Computes GPDC connectivity measures between infant-infant, adult-adult,
%    infant-adult, and adult-infant channel pairs
% 4. Averages results across frequency bands (delta, theta, alpha)

%% Initialize environment and parameters

clear all
clc

% Set base path
base_path = '/path/to/data/';

% Define analysis parameters
type_list = {'DC', 'DTF', 'PDC', 'GPDC', 'COH', 'PCOH'};
location_list = {'C', 'S'};  % C = Location 1, S = Location 2

% Process each location
for location_idx = 1:2
    % Select connectivity method (GPDC)
    connectivity_type = 4;  % 4 = GPDC
    
    % Create output directory if it doesn't exist
    output_dir = fullfile(base_path, 'data_matfile', [type_list{connectivity_type}, '3_nonorpdc_nonorpower/']);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Clear variables from previous iterations
    clearvars -except location_idx connectivity_type type_list location_list base_path output_dir
    
    % Set current location and parameters
    current_location = location_list{location_idx};
    montage = 'GRID';
    
    % Define frequency points of interest (for nfft = 256)
    freqs = [9, 17, 24, 32];  % Corresponding to ~3, 6.25, 9, 12.1 Hz
    
    % Set file paths based on location
    if strcmp(current_location, 'S')
        filepath = fullfile(base_path, 'Preprocessed_Data_location2/');
        % Location 2 participants (excluding problematic ones)
        participant_list = {'101', '104', '106', '107', '108', '110', '114', '115', ...
                           '116', '117', '120', '121', '122', '123', '127'};
    else
        filepath = fullfile(base_path, 'Preprocessed_Data_location1/');
        % Location 1 participants
        participant_list = {'101', '102', '103', '104', '105', '106', '107', '108', '109', ...
                           '111', '114', '117', '118', '119', '121', '122', '123', '124', ...
                           '125', '126', '127', '128', '129', '131', '132', '133', '135'};
    end
    
    %% Define EEG channels to include
    
    if strcmp(montage, 'GRID')
        include = [4:6, 15:17, 26:28]';  % 9-channel grid: frontal, central, parietal
        chLabel = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
    elseif strcmp(montage, 'AP3')
        include = [5, 16, 27]';  % Anterior-Posterior Midline
    elseif strcmp(montage, 'AP4')
        include = [4, 6, 26, 28]';  % Anterior-Posterior Left-Right (square)
    elseif strcmp(montage, 'CM3')
        include = [15:17]';  % Central Motor Midline
    elseif strcmp(montage, 'APM4')
        include = [15, 17, 5, 27]';  % Anterior-Posterior Midline (diamond)
    elseif strcmp(montage, 'FA3')
        include = [4:6]';  % Frontal Left-Right
    elseif strcmp(montage, 'PR3')
        include = [26, 27, 28]';  % Parietal [P3, Pz, P4]
    end
    
    %% Set MVAR model parameters
    
    MO = 7;  % Model Order (7 for 1.5s epochs)
    idMode = 7;  % MVAR method
    NSamp = 200;  % Sampling rate (Hz)
    len = 1.5;  % Window length in seconds
    shift = 0.5 * len * NSamp;  % Overlap between windows (50%)
    wlen = len * NSamp;  % Window length in samples
    nfft = 256;  % FFT size for spectral analysis
    
    fprintf('Processing location %s with %d participants...\n', current_location, length(participant_list));
    
    %% Process each participant
    
    for p = 1:length(participant_list)
        fprintf('Processing participant %d/%d: P%s%s\n', p, length(participant_list), ...
                participant_list{p}, current_location);
        
        try
            % Load EEG data
            filename = fullfile(filepath, ['P', participant_list{p}, current_location], ...
                               ['P', participant_list{p}, current_location, '_BABBLE_AR.mat']);
            
            if exist(filename, 'file')
                load(filename, 'FamEEGart', 'StimEEGart');
            else
                fprintf('Warning: File not found for participant %s\n', participant_list{p});
                continue;
            end
            
            % Initialize GPDC storage
            GPDC = cell(size(FamEEGart, 1), 1);
            for block = 1:size(FamEEGart, 1)
                if ~isempty(FamEEGart{block})
                    GPDC{block} = cell(size(FamEEGart{block}, 1), size(FamEEGart{block}, 2));
                end
            end
            
            % Initialize spectral analysis storage
            infant_peak_delta = cell(size(FamEEGart));
            infant_peak_theta = cell(size(FamEEGart));
            infant_peak_alpha = cell(size(FamEEGart));
            infant_power_delta = cell(size(FamEEGart));
            infant_power_theta = cell(size(FamEEGart));
            infant_power_alpha = cell(size(FamEEGart));
            
            adult_peak_delta = cell(size(FamEEGart));
            adult_peak_theta = cell(size(FamEEGart));
            adult_peak_alpha = cell(size(FamEEGart));
            adult_power_delta = cell(size(FamEEGart));
            adult_power_theta = cell(size(FamEEGart));
            adult_power_alpha = cell(size(FamEEGart));
            
            %% Process each block, condition, and phrase
            
            for block = 1:size(FamEEGart, 1)
                if isempty(FamEEGart{block})
                    continue;
                end
                
                for cond = 1:size(FamEEGart{block}, 1)
                    for phrase = 1:size(FamEEGart{block}, 2)
                        if ~isempty(FamEEGart{block}{cond, phrase}) && ...
                           ~isempty(StimEEGart{block}{cond, phrase}) && ...
                           size(FamEEGart{block}{cond, phrase}, 1) > 1
                            
                            % Extract infant and adult EEG data
                            infant_eeg = FamEEGart{block}{cond, phrase}(:, include);
                            adult_eeg = StimEEGart{block}{cond, phrase}(:, include);
                            
                            % Combine adult and infant data (adult first, then infant)
                            combined_eeg = [adult_eeg, infant_eeg];
                            
                            % Initialize GPDC storage for this condition/phrase
                            GPDC{block}{cond, phrase} = NaN(length(include)*2, length(include)*2, nfft, 1000);
                            
                            % Initialize spectral storage
                            infant_peak_delta{block}{cond, phrase} = [];
                            infant_peak_theta{block}{cond, phrase} = [];
                            infant_peak_alpha{block}{cond, phrase} = [];
                            infant_power_delta{block}{cond, phrase} = [];
                            infant_power_theta{block}{cond, phrase} = [];
                            infant_power_alpha{block}{cond, phrase} = [];
                            
                            adult_peak_delta{block}{cond, phrase} = [];
                            adult_peak_theta{block}{cond, phrase} = [];
                            adult_peak_alpha{block}{cond, phrase} = [];
                            adult_power_delta{block}{cond, phrase} = [];
                            adult_power_theta{block}{cond, phrase} = [];
                            adult_power_alpha{block}{cond, phrase} = [];
                            
                            %% Sliding window analysis
                            
                            w = 0;  % Window counter
                            for start_sample = 1:shift:(size(combined_eeg, 1) - wlen + 1)
                                end_sample = start_sample + wlen - 1;
                                
                                % Extract current window
                                window_data = combined_eeg(start_sample:end_sample, :);
                                
                                % Check for artifacts (NaN values)
                                if any(isnan(window_data(:)))
                                    continue;
                                end
                                
                                w = w + 1;
                                
                                %% Spectral analysis for infant channels
                                
                                infant_window = window_data(:, (length(include)+1):end);
                                
                                for ch = 1:length(include)
                                    % Compute power spectral density
                                    [Pxx, F] = pwelch(infant_window(:, ch), hanning(wlen), [], nfft, NSamp);
                                    
                                    % Delta band (1-3 Hz)
                                    delta_range = intersect(find(F >= 1), find(F <= 3));
                                    [peak_val, peak_idx] = max(Pxx(delta_range));
                                    infant_peak_delta{block}{cond, phrase} = [infant_peak_delta{block}{cond, phrase}, F(delta_range(peak_idx))];
                                    infant_power_delta{block}{cond, phrase} = [infant_power_delta{block}{cond, phrase}, mean(Pxx(delta_range))];
                                    
                                    % Theta band (3-6 Hz)
                                    theta_range = intersect(find(F >= 3), find(F <= 6));
                                    [peak_val, peak_idx] = max(Pxx(theta_range));
                                    infant_peak_theta{block}{cond, phrase} = [infant_peak_theta{block}{cond, phrase}, F(theta_range(peak_idx))];
                                    infant_power_theta{block}{cond, phrase} = [infant_power_theta{block}{cond, phrase}, mean(Pxx(theta_range))];
                                    
                                    % Alpha band (6-9 Hz)
                                    alpha_range = intersect(find(F >= 6), find(F <= 9));
                                    [peak_val, peak_idx] = max(Pxx(alpha_range));
                                    infant_peak_alpha{block}{cond, phrase} = [infant_peak_alpha{block}{cond, phrase}, F(alpha_range(peak_idx))];
                                    infant_power_alpha{block}{cond, phrase} = [infant_power_alpha{block}{cond, phrase}, mean(Pxx(alpha_range))];
                                end
                                
                                %% Spectral analysis for adult channels
                                
                                adult_window = window_data(:, 1:length(include));
                                
                                for ch = 1:length(include)
                                    % Compute power spectral density
                                    [Pxx, F] = pwelch(adult_window(:, ch), hanning(wlen), [], nfft, NSamp);
                                    
                                    % Delta band (1-3 Hz)
                                    delta_range = intersect(find(F >= 1), find(F <= 3));
                                    [peak_val, peak_idx] = max(Pxx(delta_range));
                                    adult_peak_delta{block}{cond, phrase} = [adult_peak_delta{block}{cond, phrase}, F(delta_range(peak_idx))];
                                    adult_power_delta{block}{cond, phrase} = [adult_power_delta{block}{cond, phrase}, mean(Pxx(delta_range))];
                                    
                                    % Theta band (3-6 Hz)
                                    theta_range = intersect(find(F >= 3), find(F <= 6));
                                    [peak_val, peak_idx] = max(Pxx(theta_range));
                                    adult_peak_theta{block}{cond, phrase} = [adult_peak_theta{block}{cond, phrase}, F(theta_range(peak_idx))];
                                    adult_power_theta{block}{cond, phrase} = [adult_power_theta{block}{cond, phrase}, mean(Pxx(theta_range))];
                                    
                                    % Alpha band (6-9 Hz)
                                    alpha_range = intersect(find(F >= 6), find(F <= 9));
                                    [peak_val, peak_idx] = max(Pxx(alpha_range));
                                    adult_peak_alpha{block}{cond, phrase} = [adult_peak_alpha{block}{cond, phrase}, F(alpha_range(peak_idx))];
                                    adult_power_alpha{block}{cond, phrase} = [adult_power_alpha{block}{cond, phrase}, mean(Pxx(alpha_range))];
                                end
                                
                                %% MVAR modeling and GPDC calculation
                                
                                % Transpose data for MVAR toolbox (channels × time)
                                mvar_data = window_data';
                                
                                try
                                    % Estimate MVAR model
                                    [eAm, eSu, Yp, Up] = idMVAR(mvar_data, MO, idMode);
                                    
                                    % Calculate GPDC
                                    [~, ~, ~, gpdc_result, ~, ~, ~, h, s, pp, f] = fdMVAR(eAm, eSu, nfft, NSamp);
                                    
                                    % Store GPDC result
                                    GPDC{block}{cond, phrase}(:, :, :, w) = gpdc_result;
                                    
                                catch ME
                                    % Handle MVAR estimation failures
                                    fprintf('MVAR estimation failed for P%s%s, Block %d, Cond %d, Phrase %d, Window %d: %s\n', ...
                                           participant_list{p}, current_location, block, cond, phrase, w, ME.message);
                                    GPDC{block}{cond, phrase}(:, :, :, w) = NaN(length(include)*2, length(include)*2, nfft);
                                end
                            end
                        end
                    end
                end
            end
            
            %% Average GPDC across windows and organize into quadrants
            
            % Initialize averaged GPDC storage
            avGPDC = cell(size(FamEEGart));
            
            for block = 1:size(FamEEGart, 1)
                if isempty(FamEEGart{block})
                    continue;
                end
                
                for cond = 1:size(FamEEGart{block}, 1)
                    for phrase = 1:size(FamEEGart{block}, 2)
                        if ~isempty(GPDC{block}) && ~isempty(GPDC{block}{cond, phrase}) && ...
                           size(GPDC{block}{cond, phrase}, 4) > 1
                            
                            % Average across windows (take squared magnitude for power)
                            avg_gpdc = zeros(size(GPDC{block}{cond, phrase}, 1), ...
                                           size(GPDC{block}{cond, phrase}, 2), ...
                                           size(GPDC{block}{cond, phrase}, 3));
                            
                            for i = 1:(length(include)*2)
                                for j = 1:(length(include)*2)
                                    avg_gpdc(i, j, :) = nanmean(abs(GPDC{block}{cond, phrase}(i, j, :, :)).^2, 4);
                                end
                            end
                            
                            avGPDC{block}{cond, phrase} = avg_gpdc;
                        else
                            avGPDC{block}{cond, phrase} = NaN;
                        end
                    end
                end
            end
            
            %% Split connectivity matrix into four quadrants
            
            % Initialize quadrant storage
            GPDC_II = cell(size(avGPDC));  % Infant-to-Infant
            GPDC_AA = cell(size(avGPDC));  % Adult-to-Adult
            GPDC_AI = cell(size(avGPDC));  % Adult-to-Infant
            GPDC_IA = cell(size(avGPDC));  % Infant-to-Adult
            
            for block = 1:size(avGPDC, 1)
                for cond = 1:size(avGPDC, 2)
                    for phrase = 1:size(avGPDC, 3)
                        if ~isempty(avGPDC{block, cond, phrase}) && ~all(isnan(avGPDC{block, cond, phrase}(:)))
                            tmp = avGPDC{block, cond, phrase};
                            
                            % Split the connectivity matrix into four quadrants
                            GPDC_AA{block, cond, phrase} = tmp(1:length(include), 1:length(include), :);                   % Adult-to-Adult
                            GPDC_IA{block, cond, phrase} = tmp(1:length(include), length(include)+1:length(include)*2, :); % Infant-to-Adult
                            GPDC_AI{block, cond, phrase} = tmp(length(include)+1:length(include)*2, 1:length(include), :); % Adult-to-Infant
                            GPDC_II{block, cond, phrase} = tmp(length(include)+1:length(include)*2, length(include)+1:length(include)*2, :); % Infant-to-Infant
                        else
                            GPDC_AA{block, cond, phrase} = NaN;
                            GPDC_IA{block, cond, phrase} = NaN;
                            GPDC_AI{block, cond, phrase} = NaN;
                            GPDC_II{block, cond, phrase} = NaN;
                        end
                    end
                end
            end
            
            %% Calculate average GPDC in each frequency band
            
            % Define frequency bands (indices in the nfft array)
            bands = {[4:8], [9:16], [17:24]};  % Delta (1-3 Hz), Theta (3-6 Hz), Alpha (6-9 Hz)
            
            % Initialize final storage
            II = cell(size(avGPDC, 1), size(avGPDC, 2), length(bands));
            AA = cell(size(avGPDC, 1), size(avGPDC, 2), length(bands));
            AI = cell(size(avGPDC, 1), size(avGPDC, 2), length(bands));
            IA = cell(size(avGPDC, 1), size(avGPDC, 2), length(bands));
            
            for block = 1:size(avGPDC, 1)
                for cond = 1:size(avGPDC, 2)
                    % Initialize arrays to accumulate data across phrases
                    tmp1All = [];  % For II
                    tmp2All = [];  % For AA
                    tmp3All = [];  % For AI
                    tmp4All = [];  % For IA
                    count = 0;     % Counter for valid phrases
                    
                    for phrase = 1:size(avGPDC, 3)
                        if ~isempty(avGPDC{block, cond, phrase}) && ~all(isnan(avGPDC{block, cond, phrase}(:)))
                            for fplot = 1:length(bands)
                                % Average GPDC within frequency band
                                tmp1g = squeeze(nanmean(GPDC_II{block, cond, phrase}(:, :, bands{fplot}), 3));
                                tmp2g = squeeze(nanmean(GPDC_AA{block, cond, phrase}(:, :, bands{fplot}), 3));
                                tmp3g = squeeze(nanmean(GPDC_AI{block, cond, phrase}(:, :, bands{fplot}), 3));
                                tmp4g = squeeze(nanmean(GPDC_IA{block, cond, phrase}(:, :, bands{fplot}), 3));
                                
                                % Initialize storage arrays if needed
                                if isempty(tmp1All)
                                    tmp1All = zeros([size(tmp1g), length(bands)]);
                                    tmp2All = zeros([size(tmp2g), length(bands)]);
                                    tmp3All = zeros([size(tmp3g), length(bands)]);
                                    tmp4All = zeros([size(tmp4g), length(bands)]);
                                end
                                
                                % Accumulate data
                                tmp1All(:, :, fplot) = tmp1All(:, :, fplot) + tmp1g;
                                tmp2All(:, :, fplot) = tmp2All(:, :, fplot) + tmp2g;
                                tmp3All(:, :, fplot) = tmp3All(:, :, fplot) + tmp3g;
                                tmp4All(:, :, fplot) = tmp4All(:, :, fplot) + tmp4g;
                            end
                            count = count + 1;
                        end
                    end
                    
                    % Calculate averages if valid data exists
                    if count > 0
                        for fplot = 1:length(bands)
                            II{block, cond, fplot} = tmp1All(:, :, fplot) / count;
                            AA{block, cond, fplot} = tmp2All(:, :, fplot) / count;
                            AI{block, cond, fplot} = tmp3All(:, :, fplot) / count;
                            IA{block, cond, fplot} = tmp4All(:, :, fplot) / count;
                        end
                    end
                end
            end
            
            % Clear intermediate variables
            clear GPDC_II GPDC_AA GPDC_AI GPDC_IA avGPDC GPDC
            
            %% Save GPDC matrices
            
            % Save connectivity matrices
            if strcmp(current_location, 'C')
                output_file = fullfile(output_dir, ['UK_', participant_list{p}, '_PDC.mat']);
            else
                output_file = fullfile(output_dir, ['SG_', participant_list{p}, '_PDC.mat']);
            end
            
            save(output_file, 'II', 'AI', 'AA', 'IA');
            
            fprintf('Connectivity analysis complete and saved for participant %s\n', participant_list{p});
            
        catch ME
            fprintf('Error processing participant %s: %s\n', participant_list{p}, ME.message);
        end
        
    end  % End of participant loop
    
    fprintf('Completed processing for location %s\n', current_location);
    
end  % End of location loop

%% Summary of Analysis Approach
% 
% This script performs the following key steps:
%
% 1. Preprocesses EEG data from adult-infant dyads across multiple conditions
%
% 2. Calculates GPDC (Generalized Partial Directed Coherence) using MVAR modeling:
%    - Segments data into overlapping windows (1.5s with 50% overlap)
%    - Estimates MVAR model parameters for each window
%    - Computes frequency-domain connectivity measures
%    - Squares magnitude to represent power contributions
%
% 3. Organizes connectivity into four quadrants:
%    - II: Infant-to-Infant connectivity (within-brain)
%    - AA: Adult-to-Adult connectivity (within-brain)
%    - AI: Adult-to-Infant connectivity (between-brain)
%    - IA: Infant-to-Adult connectivity (between-brain)
%
% 4. Averages connectivity across three frequency bands:
%    - Delta: 1-3 Hz
%    - Theta: 3-6 Hz
%    - Alpha: 6-9 Hz
%
% These connectivity measures can be used to assess how different experimental
% conditions affect neural synchrony between infants and adults during social
% interaction.

fprintf('\nGPDC connectivity analysis complete.\n');
