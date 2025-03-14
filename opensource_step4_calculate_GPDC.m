%% EEG Connectivity Analysis with GPDC
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.
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

% Set base path
base_path = '/path/to/data/';

% Define analysis parameters
type_list = {'DC', 'DTF', 'PDC', 'GPDC', 'COH', 'PCOH'};
location_list = {'C', 'S'};  % C = Location 1, S = Location 2

% Process each location
for location_idx = 1:2
    % Select connectivity method (GPDC)
    connectivity_type = 4;
    
    % Create output directory if it doesn't exist
    output_dir = fullfile(base_path, 'data_matfile', [type_list{connectivity_type}, '*/']);
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
    else
        filepath = fullfile(base_path, 'Preprocessed_Data_location1/');
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
    idMode = 7;  % MVAR method (7 = Nuttall-Strand unbiased partial correlation)
    fs = 200;  % Sampling rate (Hz)
    len = 1.5;  % Epoch length in seconds
    window_length = len * fs;  % Window length in samples
    shift = 0.5 * window_length;  % 50% overlap between windows
    nfft = 256;  % FFT size
    
    %% Load participant lists
    
    if strcmp(current_location, 'S')
        participant_list = {'101', '104', '106', '107', '108', '110', '114', '115', 
                           '116', '117', '120', '121', '122', '123', '127'};
    else
        participant_list = {'101', '102', '103', '104', '105', '106', '107', '108', '109', 
                           '111', '114', '117', '118', '119', '121', '122', '123', '124', 
                           '125', '126', '127', '128', '129', '131', '132', '133', '135'};
    end
    
    %% Initialize data structure for connectivity analysis
    
    % Pre-allocate cell array for GPDC results
    avGPDC = cell(length(participant_list), 1);
    
    %% Process each participant
    
    for p = 1:length(participant_list)
        % Load preprocessed EEG data
        filename = fullfile(filepath, ['P', participant_list{p}, current_location], 
                           ['P', participant_list{p}, current_location, '*_AR.mat']);
        loaded_data = load(filename, 'FamEEGart', 'StimEEGart');
        FamEEGart = loaded_data.FamEEGart;  % Infant EEG
        StimEEGart = loaded_data.StimEEGart;  % Adult EEG
        
        % Extract selected channels and mark bad segments
        iEEG = cell(size(FamEEGart));
        aEEG = cell(size(StimEEGart));
        
        for block = 1:size(FamEEGart, 1)
            if ~isempty(FamEEGart{block})
                for cond = 1:size(FamEEGart{block}, 1)
                    for phrase = 1:size(FamEEGart{block}, 2)
                        if ~isempty(FamEEGart{block}{cond, phrase}) && size(FamEEGart{block}{cond, phrase}, 1) > 1
                            for chan = 1:length(include)
                                % Extract channels of interest
                                iEEG{block}{cond, phrase}(:, chan) = FamEEGart{block}{cond, phrase}(:, include(chan));
                                aEEG{block}{cond, phrase}(:, chan) = StimEEGart{block}{cond, phrase}(:, include(chan));
                                
                                % Mark bad segments (777=unattended, 888=manual rejection, 999=auto rejection)
                                bad_idx = find(FamEEGart{block}{cond, phrase}(:, chan) == 777 | 
                                              FamEEGart{block}{cond, phrase}(:, chan) == 888 | 
                                              FamEEGart{block}{cond, phrase}(:, chan) == 999);
                                
                                if ~isempty(bad_idx)
                                    iEEG{block}{cond, phrase}(bad_idx, chan) = 1000;  % Mark as bad
                                    aEEG{block}{cond, phrase}(bad_idx, chan) = 1000;  % Mark as bad
                                end
                            end
                        end
                    end
                end
            else
                iEEG{block} = [];
                aEEG{block} = [];
            end
        end
        
        % Combine adult and infant EEG data for joint analysis
        EEG = cell(size(FamEEGart, 1), 1);
        for block = 1:size(FamEEGart, 1)
            if ~isempty(iEEG{block})
                for cond = 1:size(FamEEGart{block}, 1)
                    for phrase = 1:size(FamEEGart{block}, 2)
                        if ~isempty(iEEG{block}{cond, phrase})
                            EEG{block}{cond, phrase} = horzcat(aEEG{block}{cond, phrase}, iEEG{block}{cond, phrase});
                        end
                    end
                end
            end
        end
        
        %% Compute windowed GPDC and spectral measures
        
        for block = 1:size(EEG, 1)
            if ~isempty(EEG{block})
                for cond = 1:size(EEG{block}, 1)
                    for phrase = 1:size(EEG{block}, 2)
                        if ~isempty(EEG{block}{cond, phrase}) && size(EEG{block}{cond, phrase}, 1) > 1
                            % Calculate number of windows
                            n_windows = floor((size(EEG{block}{cond, phrase}, 1) - window_length) / shift) + 1;
                            
                            % Process each window
                            for w = 1:n_windows
                                % Extract window of data
                                window_data = EEG{block}{cond, phrase}((w-1)*shift+1:(w-1)*shift+window_length, :);
                                
                                % Process only if no bad data in window
                                if ~any(window_data(:) == 1000) && ~any(isnan(window_data(:)))
                                    % MVAR analysis focuses on connectivity, not spectral power
                                    
                                    % MVAR analysis for connectivity
                                    % Transpose data for MVAR analysis
                                    mvar_data = window_data';
                                    
                                    % Estimate MVAR model
                                    [eAm, eSu, Yp, Up] = idMVAR(mvar_data, MO, idMode);
                                    
                                    % Calculate frequency-domain measures
                                    if connectivity_type == 1
                                        [GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~, ~, h, s, pp, f] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 2
                                        [~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~, h, s, pp, f] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 3
                                        [~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, h, s, pp, f] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 4
                                        [~, ~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, h, s, pp, f] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 5
                                        [~, ~, ~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, h, s, pp, f] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 6
                                        [~, ~, ~, ~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, h, s, pp, f] = fdMVAR(eAm, eSu, nfft, fs);
                                    end
                                else
                                    GPDC{block}{cond, phrase}(:,:,:,w) = NaN(length(include)*2, length(include)*2, nfft);
                                end
                            end
                        else
                            GPDC{block}{cond, phrase} = [];
                        end
                    end
                end
            else
                GPDC{block} = [];
            end
        end
        
        %% Average GPDC over windows
        
        for block = 1:size(GPDC, 2)
            for cond = 1:size(GPDC{block}, 1)
                for phrase = 1:size(GPDC{block}, 2)
                    if ~isempty(GPDC{block}{cond, phrase}) && length(GPDC{block}{cond, phrase}) > 1
                        for i = 1:2*length(include)
                            for j = 1:2*length(include)
                                % Square magnitude for power representation
                                tmp2(i, j, :) = nanmean(abs(GPDC{block}{cond, phrase}(i, j, :, :)).^2, 4);
                            end
                        end
                        avGPDC{p, 1}{block, cond, phrase} = tmp2;
                        clear tmp2
                    else
                        avGPDC{p, 1}{block, cond, phrase} = NaN;
                    end
                end
            end
        end
        
        % Clear large variables to free memory
        clear FamEEGart StimEEGart iEEG aEEG EEG GPDC
        
        fprintf('Processed participant %d of %d\n', p, length(participant_list));
    end
    
    %% Extract connectivity matrices for each participant
    
    for p = 1:length(participant_list)
        fprintf('Extracting connectivity matrices for participant %d\n', p);
        
        II = {};  % Infant-to-Infant connectivity
        AA = {};  % Adult-to-Adult connectivity
        AI = {};  % Adult-to-Infant connectivity
        IA = {};  % Infant-to-Adult connectivity
        
        for block = 1:size(avGPDC{p}, 1)
            for cond = 1:size(avGPDC{p}, 2)
                for phase = 1:size(avGPDC{p}, 3)
                    if ~isempty(avGPDC{p}{block, cond, phase}) && nansum(nansum(nansum(avGPDC{p}{block, cond, phase}))) ~= 0
                        tmp = avGPDC{p}{block, cond, phase};
                        
                        % Split the connectivity matrix into four quadrants
                        GPDC_AA{block, cond, phase} = tmp(1:length(include), 1:length(include), :);                   % Adult-to-Adult
                        GPDC_IA{block, cond, phase} = tmp(1:length(include), length(include)+1:length(include)*2, :); % Infant-to-Adult
                        GPDC_AI{block, cond, phase} = tmp(length(include)+1:length(include)*2, 1:length(include), :); % Adult-to-Infant
                        GPDC_II{block, cond, phase} = tmp(length(include)+1:length(include)*2, length(include)+1:length(include)*2, :); % Infant-to-Infant
                    else
                        GPDC_AA{block, cond, phase} = NaN;
                        GPDC_IA{block, cond, phase} = NaN;
                        GPDC_AI{block, cond, phase} = NaN;
                        GPDC_II{block, cond, phase} = NaN;
                    end
                end
            end
        end
        
        %% Calculate average GPDC in each frequency band
        
        % Define frequency bands (indices in the nfft array)
        bands = {[4:8], [9:16], [17:24]};  % Delta (1-3 Hz), Theta (3-6 Hz), Alpha (6-9 Hz)
        
        for block = 1:size(avGPDC{p}, 1)
            for cond = 1:size(avGPDC{p}, 2)
                % Initialize arrays to accumulate data across phases
                tmp1All = [];  % For II
                tmp2All = [];  % For AA
                tmp3All = [];  % For AI
                tmp4All = [];  % For IA
                count = 0;     % Counter for valid phases
                
                for phase = 1:size(avGPDC{p}, 3)
                    if ~isempty(avGPDC{p}{block, cond, phase}) && nansum(nansum(nansum(avGPDC{p}{block, cond, phase}))) ~= 0
                        for fplot = 1:length(bands)
                            % Average GPDC within frequency band
                            tmp1g = squeeze(nanmean(GPDC_II{block, cond, phase}(:, :, bands{fplot}), 3));
                            tmp2g = squeeze(nanmean(GPDC_AA{block, cond, phase}(:, :, bands{fplot}), 3));
                            tmp3g = squeeze(nanmean(GPDC_AI{block, cond, phase}(:, :, bands{fplot}), 3));
                            tmp4g = squeeze(nanmean(GPDC_IA{block, cond, phase}(:, :, bands{fplot}), 3));
                            
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
        clear GPDC_II GPDC_AA GPDC_AI GPDC_IA
        
        %% Save GPDC matrices
        
        % Save connectivity matrices
        output_file = fullfile(output_dir, sprintf('%s_%s_PDC.mat', 
                              current_location == 'C' ? 'UK' : 'SG', participant_list{p}));
        save(output_file, 'II', 'AI', 'AA', 'IA');
        
        fprintf('Connectivity analysis complete and saved for participant %s\n', participant_list{p});
        
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
%    - Segments data into overlapping windows
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