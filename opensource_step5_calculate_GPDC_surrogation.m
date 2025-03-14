%% EEG Surrogate Connectivity Analysis
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.
%
% Purpose: Generate surrogate data by shuffling epochs within each participant,
% then calculate PDC (Partial Directed Coherence) to establish null distributions
% for statistical testing
%
% This script:
% 1. Loads preprocessed EEG data from two locations
% 2. Creates surrogate data by shuffling windows across conditions while preserving 
%    within-participant characteristics
% 3. Computes connectivity measures on surrogate data
% 4. Saves results for comparison with actual connectivity patterns

%% Initialize environment
clc
clear all

% Set base path
base_path = '/path/to/data/';

% Define connectivity method and locations
connectivity_type = 3;  % 3 = PDC (Partial Directed Coherence)
type_list = {'DC', 'DTF', 'PDC', 'GPDC', 'COH', 'PCOH'};
location_list = {'C', 'S'};  % C = Location 1, S = Location 2

% Set surrogate parameters
surrogate_set = 5;  % Current surrogate set number
surrogate_indices = 736:740;  % Range of surrogate iterations to process

% Check surrogate file structure
surrogate_path_prefix = fullfile(base_path, 'data_matfile', ['surrGPDCSET', num2str(surrogate_set), '/']);
file_counts = zeros(1000, 1);
for surr_idx = 1:1000
    surr_path = fullfile(surrogate_path_prefix, num2str(surr_idx));
    files = dir(fullfile(surr_path, '*.mat'));
    file_counts(surr_idx) = length(files);
end

%% Load EEG data from both locations

% Load Location 2 data
location2_path = fullfile(base_path, 'Preprocessed_Data_location2/');
location2_participants = {'101', '104', '106', '107', '108', '110', '114', '115', 
                        '116', '117', '120', '121', '122', '123', '127'};

% Create data structure for Location 2
location2_data = struct();
for p = 1:length(location2_participants)
    filename = fullfile(location2_path, ['P', location2_participants{p}, 'S'], 
                       ['P', location2_participants{p}, 'S', '*_AR.mat']);
    loaded_data = load(filename, 'FamEEGart', 'StimEEGart');
    location2_data(p).FamEEGart = loaded_data.FamEEGart;
    location2_data(p).StimEEGart = loaded_data.StimEEGart;
end

% Load Location 1 data
location1_path = fullfile(base_path, 'Preprocessed_Data_location1/');
location1_participants = {'101', '102', '103', '104', '105', '106', '107', '108', '109', 
                        '111', '114', '117', '118', '119', '121', '122', '123', '124', 
                        '125', '126', '127', '128', '129', '131', '132', '133', '135'};

% Create data structure for Location 1
location1_data = struct();
for p = 1:length(location1_participants)
    filename = fullfile(location1_path, ['P', location1_participants{p}, 'C'], 
                       ['P', location1_participants{p}, 'C', '*_AR.mat']);
    loaded_data = load(filename, 'FamEEGart', 'StimEEGart');
    location1_data(p).FamEEGart = loaded_data.FamEEGart;
    location1_data(p).StimEEGart = loaded_data.StimEEGart;
end

%% Process surrogate iterations
for surr_idx = surrogate_indices
    % Skip if already processed enough files
    if file_counts(surr_idx) > 83
        continue;
    end
    
    % Create output directory if it doesn't exist
    output_dir = fullfile(base_path, 'data_matfile', ['surrPDCSET', num2str(surrogate_set)], 
                         ['PDC', num2str(surr_idx)]);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Process each location
    for location_idx = 1:2
        % Current location
        current_location = location_list{location_idx};
        
        % Clear variables from previous iterations (except those needed)
        clearvars -except location_idx connectivity_type type_list location_list surr_idx ...
                         surrogate_set base_path output_dir location1_data location2_data ...
                         location1_participants location2_participants file_counts surrogate_indices
        
        %% Set analysis parameters
        
        % Define montage and channels
        montage = 'GRID';
        include = [4:6, 15:17, 26:28]';  % 9-channel grid covering frontal, central, parietal
        chLabel = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
        
        % Set MVAR model parameters
        MO = 7;  % Model order (for 1.5s epochs)
        idMode = 7;  % MVAR estimation method (Nuttall-Strand)
        fs = 200;  % Sampling rate in Hz
        len = 1.5;  % Epoch length in seconds
        window_length = len * fs;  % Window length in samples
        shift = 0.5 * window_length;  % 50% overlap between windows
        nfft = 256;  % FFT size
        
        % Select participant list based on location
        if strcmp(current_location, 'S')
            participant_list = location2_participants;
        else
            participant_list = location1_participants;
        end
        
        %% Process each participant
        
        for p = 1:length(participant_list)
            % Load EEG data for current participant
            if location_idx == 1
                FamEEGart = location1_data(p).FamEEGart;  % Infant EEG
                StimEEGart = location1_data(p).StimEEGart;  % Adult EEG
            else
                FamEEGart = location2_data(p).FamEEGart;  % Infant EEG
                StimEEGart = location2_data(p).StimEEGart;  % Adult EEG
            end
            
            % Initialize arrays for EEG data
            iEEG = cell(size(FamEEGart));
            aEEG = cell(size(StimEEGart));
            
            % Extract selected channels and mark bad segments
            for block = 1:size(FamEEGart, 1)
                if ~isempty(FamEEGart{block})
                    for cond = 1:size(FamEEGart{block}, 1)
                        for phrase = 1:size(FamEEGart{block}, 2)
                            if ~isempty(FamEEGart{block}{cond, phrase}) && size(FamEEGart{block}{cond, phrase}, 1) > 1
                                for chan = 1:length(include)
                                    % Extract channels of interest
                                    iEEG{block}{cond, phrase}(:, chan) = FamEEGart{block}{cond, phrase}(:, include(chan));
                                    aEEG{block}{cond, phrase}(:, chan) = StimEEGart{block}{cond, phrase}(:, include(chan));
                                    
                                    % Mark bad segments (777=unattended, 888=manual reject, 999=auto reject)
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
            
            % Combine adult and infant EEG for joint analysis
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
            
            %% Extract windows of valid data
            
            windowlist = cell(3, 3, 3, 1);  % Initialize window storage array (block, cond, phrase, window)
            
            for block = 1:size(EEG, 1)
                if ~isempty(EEG{block})
                    for cond = 1:size(EEG{block}, 1)
                        for phrase = 1:size(EEG{block}, 2)
                            if ~isempty(EEG{block}{cond, phrase}) && size(EEG{block}{cond, phrase}, 1) > 1
                                % Calculate number of windows
                                num_windows = floor((size(EEG{block}{cond, phrase}, 1) - window_length) / shift) + 1;
                                
                                % Process each window
                                for w = 1:num_windows
                                    % Extract window of data
                                    window_data = EEG{block}{cond, phrase}((w-1)*shift+1:(w-1)*shift+window_length, :);
                                    
                                    % Keep only if no bad data in window
                                    if ~any(window_data(:) == 1000) && ~any(isnan(window_data(:)))
                                        windowlist{block, cond, phrase, w} = window_data;
                                    else
                                        windowlist{block, cond, phrase, w} = [];
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
            %% Create surrogate data by shuffling windows
            
            % Find all non-empty windows
            [dim1, dim2, dim3, dim4] = size(windowlist);
            non_empty_positions = [];
            
            % Collect positions of all valid windows
            for i = 1:dim1
                for j = 1:dim2
                    for k = 1:dim3
                        for l = 1:dim4
                            if ~isempty(windowlist{i, j, k, l}) && isequal(size(windowlist{i, j, k, l}), [window_length, 2*length(include)])
                                non_empty_positions = [non_empty_positions; [i, j, k, l]];
                            end
                        end
                    end
                end
            end
            
            % Number of valid windows
            num_non_empty_positions = size(non_empty_positions, 1);
            
            % Number of channels to shuffle
            num_channels = 2*length(include);
            
            % Generate random permutations for each channel
            random_orders = cell(1, num_channels);
            for i = 1:num_channels
                random_orders{i} = randperm(num_non_empty_positions);
            end
            
            % Create shuffled windowlist
            temp_windowlist = windowlist;
            
            % Shuffle each window
            for idx = 1:num_non_empty_positions
                % Current window position
                current_pos = non_empty_positions(idx, :);
                
                % Create shuffled window by combining channels from different windows
                shuffled_window = [];
                for i = 1:num_channels
                    % Get random window for this channel
                    random_pos = non_empty_positions(random_orders{i}(idx), :);
                    random_window = windowlist{random_pos(1), random_pos(2), random_pos(3), random_pos(4)};
                    
                    % Extract the channel
                    shuffled_window = [shuffled_window, random_window(:, i)];
                end
                
                % Store shuffled window
                temp_windowlist{current_pos(1), current_pos(2), current_pos(3), current_pos(4)} = shuffled_window;
            end
            
            % Replace with shuffled windows
            windowlist = temp_windowlist;
            
            %% Calculate MVAR-based connectivity on surrogate data
            
            GPDC = cell(size(windowlist, 1), size(windowlist, 2), size(windowlist, 3));
            
            for block = 1:size(windowlist, 1)
                for cond = 1:size(windowlist, 2)
                    for phrase = 1:size(windowlist, 3)
                        for w = 1:size(windowlist, 4)
                            if ~isempty(windowlist{block, cond, phrase, w})
                                window_data = windowlist{block, cond, phrase, w};
                                
                                % MVAR analysis focuses on connectivity
                                
                                % Transpose data for MVAR analysis
                                mvar_data = window_data';
                                
                                % Estimate MVAR model
                                try
                                    [eAm, eSu, ~, ~] = idMVAR(mvar_data, MO, idMode);
                                    
                                    % Calculate connectivity measure based on selected type
                                    if connectivity_type == 1
                                        [GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 2
                                        [~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 3
                                        [~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 4
                                        [~, ~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 5
                                        [~, ~, ~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, fs);
                                    elseif connectivity_type == 6
                                        [~, ~, ~, ~, ~, GPDC{block}{cond, phrase}(:,:,:,w), ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, fs);
                                    end
                                catch
                                    % Set to NaN if MVAR estimation fails
                                    GPDC{block}{cond, phrase}(:,:,:,w) = NaN(length(include)*2, length(include)*2, nfft);
                                end
                            else
                                GPDC{block}{cond, phrase}(:,:,:,w) = NaN(length(include)*2, length(include)*2, nfft);
                            end
                        end
                    end
                end
            end
            
            %% Average GPDC over windows
            
            avGPDC = cell(1, 1);
            
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
            
            %% Extract connectivity matrices for each participant
            
            II = {};  % Infant-to-Infant connectivity
            AA = {};  % Adult-to-Adult connectivity
            AI = {};  % Adult-to-Infant connectivity
            IA = {};  % Infant-to-Adult connectivity
            
            for block = 1:size(avGPDC{p}, 1)
                for cond = 1:size(avGPDC{p}, 2)
                    for phase = 1:size(avGPDC{p}, 3)
                        if ~isempty(avGPDC{p}{block, cond, phase}) && nansum(nansum(nansum(avGPDC{p}{block, cond, phase}))) ~= 0
                            tmp = avGPDC{p}{block, cond, phase};
                            
                            % Split connectivity matrix into four quadrants
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
            
            %% Calculate average connectivity in frequency bands
            
            % Define frequency bands (indices in nfft)
            bands = {[4:8], [9:16], [17:24]};  % Delta (1-3 Hz), Theta (3-6 Hz), Alpha (6-9 Hz)
            
            for block = 1:size(avGPDC{p}, 1)
                for cond = 1:size(avGPDC{p}, 2)
                    % Initialize arrays to accumulate data
                    tmp1All = [];  % For II
                    tmp2All = [];  % For AA
                    tmp3All = [];  % For AI
                    tmp4All = [];  % For IA
                    count = 0;     % Counter for valid phases
                    
                    for phase = 1:size(avGPDC{p}, 3)
                        if ~isempty(avGPDC{p}{block, cond, phase}) && nansum(nansum(nansum(avGPDC{p}{block, cond, phase}))) ~= 0
                            for fplot = 1:length(bands)
                                % Average connectivity within frequency band
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
            
            %% Save surrogate connectivity results
            
            % Save connectivity matrices
            save_filename = fullfile(output_dir, sprintf('%s_%s_PDC.mat', 
                                   current_location == 'C' ? 'UK' : 'SG', participant_list{p}));
            save(save_filename, 'II', 'AI', 'AA', 'IA');
            
            fprintf('Processed participant %s from location %s for surrogate %d\n', 
                  participant_list{p}, current_location, surr_idx);
        end
    end
    
    fprintf('Completed surrogate iteration %d\n', surr_idx);
end

%% Summary of Surrogate Analysis Approach
% 
% This script performs the following steps to generate null-hypothesis data:
%
% 1. Loads preprocessed EEG data from adult-infant dyads
%
% 2. Creates surrogate data using a channel shuffling approach:
%    - Extracts valid data windows across all conditions and phrases
%    - Randomly reassigns channels across windows to break true connectivity
%    - Preserves spectral characteristics of individual channels
%
% 3. Calculates PDC (Partial Directed Coherence) on surrogate data:
%    - Applies MVAR modeling to shuffled windows
%    - Computes frequency-domain connectivity measures
%    - Averages connectivity within key frequency bands
%
% 4. Organizes results into four quadrants for statistical testing:
%    - II: Infant-to-Infant connectivity
%    - AA: Adult-to-Adult connectivity
%    - AI: Adult-to-Infant connectivity
%    - IA: Infant-to-Adult connectivity
%
% These surrogate datasets provide a reference distribution for statistical testing,
% allowing assessment of whether observed connectivity patterns differ from chance.