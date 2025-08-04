%% EEG Surrogate Connectivity Analysis
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
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
clear all
clc

% Set base path
base_path = '/path/to/data/';

%% Check surrogate file structure and define processing range

% Define surrogate path prefix
surrogate_path_prefix = fullfile(base_path, 'data_matfile', 'surrGPDCSET5/');

% Check existing surrogate files
file_counts = zeros(1000, 1);
for surr_idx = 1:1000
    surr_path = fullfile(surrogate_path_prefix, num2str(surr_idx));
    files = dir(fullfile(surr_path, '*.mat'));
    file_counts(surr_idx) = length(files);
end

%% Load EEG data from both locations

fprintf('Loading EEG data from both locations...\n');

% Load Location 2 (Singapore) data
location2_path = fullfile(base_path, 'Preprocessed_Data_location2/');
location2_participants = {'101', '104', '106', '107', '108', '110', '114', '115', ...
                         '116', '117', '120', '121', '122', '123', '127'};

location2_data = struct();
for p = 1:length(location2_participants)
    filename = fullfile(location2_path, ['P', location2_participants{p}, 'S'], ...
                       ['P', location2_participants{p}, 'S', '_BABBLE_AR.mat']);
    if exist(filename, 'file')
        loaded_data = load(filename, 'FamEEGart', 'StimEEGart');
        location2_data(p).FamEEGart = loaded_data.FamEEGart;
        location2_data(p).StimEEGart = loaded_data.StimEEGart;
    else
        fprintf('Warning: File not found for Location 2 participant %s\n', location2_participants{p});
    end
end

% Load Location 1 (Cambridge) data
location1_path = fullfile(base_path, 'Preprocessed_Data_location1/');
location1_participants = {'101', '102', '103', '104', '105', '106', '107', '108', '109', ...
                         '111', '114', '117', '118', '119', '121', '122', '123', '124', ...
                         '125', '126', '127', '128', '129', '131', '132', '133', '135'};

location1_data = struct();
for p = 1:length(location1_participants)
    filename = fullfile(location1_path, ['P', location1_participants{p}, 'C'], ...
                       ['P', location1_participants{p}, 'C', '_BABBLE_AR.mat']);
    if exist(filename, 'file')
        loaded_data = load(filename, 'FamEEGart', 'StimEEGart');
        location1_data(p).FamEEGart = loaded_data.FamEEGart;
        location1_data(p).StimEEGart = loaded_data.StimEEGart;
    else
        fprintf('Warning: File not found for Location 1 participant %s\n', location1_participants{p});
    end
end

%% Process surrogate iterations

% Process surrogate iterations 736-740 (or adjust range as needed)
surrogate_range = 736:740;

for surr_idx = surrogate_range
    fprintf('\n=== Processing surrogate iteration %d ===\n', surr_idx);
    
    % Skip if already processed enough files
    if file_counts(surr_idx) > 83
        fprintf('Surrogate %d already has sufficient files (%d), skipping...\n', ...
                surr_idx, file_counts(surr_idx));
        continue;
    end
    
    % Create output directory for this surrogate iteration
    output_dir = fullfile(base_path, 'data_matfile', 'surrPDCSET5', ['PDC', num2str(surr_idx)]);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    %% Process each location
    
    type_list = {'DC', 'DTF', 'PDC', 'GPDC', 'COH', 'PCOH'};
    location_list = {'C', 'S'};  % C = Location 1, S = Location 2
    
    for location_idx = 1:2
        % Current location
        current_location = location_list{location_idx};
        
        % Select connectivity method (PDC)
        connectivity_type = 3;  % 3 = PDC
        
        fprintf('Processing location %s for surrogate %d...\n', current_location, surr_idx);
        
        %% Set analysis parameters
        
        % Define montage and channels
        montage = 'GRID';
        include = [4:6, 15:17, 26:28]';  % 9-channel grid covering frontal, central, parietal
        chLabel = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
        
        % Set frequency points of interest (for nfft = 256)
        freqs = [9, 17, 24, 32];  % Corresponding to ~3, 6.25, 9, 12.1 Hz
        
        % Set MVAR model parameters
        MO = 7;  % Model order (for 1.5s epochs)
        idMode = 7;  % MVAR estimation method (Nuttall-Strand)
        NSamp = 200;  % Sampling rate in Hz
        len = 1.5;  % Epoch length in seconds
        window_length = len * NSamp;  % Window length in samples
        shift = 0.5 * window_length;  % 50% overlap between windows
        nfft = 256;  % FFT size
        
        % Select participant list and data based on location
        if strcmp(current_location, 'S')
            participant_list = location2_participants;
            data_struct = location2_data;
        else
            participant_list = location1_participants;
            data_struct = location1_data;
        end
        
        %% Process each participant
        
        for p = 1:length(participant_list)
            fprintf('Processing participant P%s%s (%d/%d)...\n', ...
                    participant_list{p}, current_location, p, length(participant_list));
            
            try
                % Load EEG data for current participant
                FamEEGart = data_struct(p).FamEEGart;  % Infant EEG
                StimEEGart = data_struct(p).StimEEGart;  % Adult EEG
                
                if isempty(FamEEGart) || isempty(StimEEGart)
                    fprintf('Warning: Empty data for participant %s\n', participant_list{p});
                    continue;
                end
                
                %% Extract and prepare data for surrogate analysis
                
                % Initialize arrays for EEG data
                iEEG = cell(size(FamEEGart));
                aEEG = cell(size(StimEEGart));
                
                % Extract selected channels and mark bad segments
                for block = 1:size(FamEEGart, 1)
                    if ~isempty(FamEEGart{block})
                        for cond = 1:size(FamEEGart{block}, 1)
                            for phrase = 1:size(FamEEGart{block}, 2)
                                if ~isempty(FamEEGart{block}{cond, phrase}) && ...
                                   ~isempty(StimEEGart{block}{cond, phrase}) && ...
                                   size(FamEEGart{block}{cond, phrase}, 1) > 1
                                    
                                    % Extract selected channels
                                    for chan = 1:length(include)
                                        iEEG{block}{cond, phrase}(:, chan) = ...
                                            FamEEGart{block}{cond, phrase}(:, include(chan));
                                        aEEG{block}{cond, phrase}(:, chan) = ...
                                            StimEEGart{block}{cond, phrase}(:, include(chan));
                                        
                                        % Mark rejected segments with value 1000
                                        rejected_idx = find(FamEEGart{block}{cond, phrase}(:, chan) == 777 | ...
                                                          FamEEGart{block}{cond, phrase}(:, chan) == 888 | ...
                                                          FamEEGart{block}{cond, phrase}(:, chan) == 999);
                                        if ~isempty(rejected_idx)
                                            iEEG{block}{cond, phrase}(rejected_idx, chan) = 1000;
                                            aEEG{block}{cond, phrase}(rejected_idx, chan) = 1000;
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
                
                %% Collect all valid data windows for shuffling
                
                all_windows = [];
                window_info = [];  % Store [block, cond, phrase, window_start] info
                
                for block = 1:size(FamEEGart, 1)
                    if ~isempty(iEEG{block})
                        for cond = 1:size(iEEG{block}, 1)
                            for phrase = 1:size(iEEG{block}, 2)
                                if ~isempty(iEEG{block}{cond, phrase}) && ...
                                   ~isempty(aEEG{block}{cond, phrase})
                                    
                                    % Combine adult and infant data (adult first, then infant)
                                    combined_eeg = [aEEG{block}{cond, phrase}, iEEG{block}{cond, phrase}];
                                    
                                    % Extract overlapping windows
                                    for start_sample = 1:shift:(size(combined_eeg, 1) - window_length + 1)
                                        end_sample = start_sample + window_length - 1;
                                        window_data = combined_eeg(start_sample:end_sample, :);
                                        
                                        % Check for artifacts (skip windows with NaN or rejection codes)
                                        if ~any(isnan(window_data(:))) && ~any(window_data(:) == 1000)
                                            all_windows = cat(3, all_windows, window_data);
                                            window_info = [window_info; [block, cond, phrase, start_sample]];
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                
                if isempty(all_windows)
                    fprintf('Warning: No valid windows found for participant %s\n', participant_list{p});
                    continue;
                end
                
                %% Create surrogate data by shuffling windows
                
                % Randomly shuffle the assignment of windows to conditions/phrases
                num_windows = size(all_windows, 3);
                shuffled_indices = randperm(num_windows);
                
                % Reassign windows to original structure with shuffled data
                GPDC = cell(size(FamEEGart, 1), 1);
                for block = 1:size(FamEEGart, 1)
                    if ~isempty(FamEEGart{block})
                        GPDC{block} = cell(size(FamEEGart{block}, 1), size(FamEEGart{block}, 2));
                    end
                end
                
                window_counter = 1;
                
                %% Calculate GPDC on surrogate data
                
                for block = 1:size(FamEEGart, 1)
                    if ~isempty(FamEEGart{block})
                        for cond = 1:size(FamEEGart{block}, 1)
                            for phrase = 1:size(FamEEGart{block}, 2)
                                if ~isempty(FamEEGart{block}{cond, phrase})
                                    
                                    % Initialize GPDC storage
                                    GPDC{block}{cond, phrase} = NaN(length(include)*2, length(include)*2, nfft, 1000);
                                    
                                    % Find how many windows this condition/phrase should have
                                    original_windows = sum(window_info(:,1) == block & ...
                                                         window_info(:,2) == cond & ...
                                                         window_info(:,3) == phrase);
                                    
                                    if original_windows > 0
                                        w = 0;  % Window counter for this condition/phrase
                                        
                                        for win = 1:original_windows
                                            if window_counter <= num_windows
                                                % Get shuffled window
                                                shuffled_window = all_windows(:, :, shuffled_indices(window_counter));
                                                window_counter = window_counter + 1;
                                                w = w + 1;
                                                
                                                % Transpose data for MVAR toolbox (channels × time)
                                                mvar_data = shuffled_window';
                                                
                                                try
                                                    % Estimate MVAR model
                                                    [eAm, eSu, Yp, Up] = idMVAR(mvar_data, MO, idMode);
                                                    
                                                    % Calculate PDC based on connectivity type
                                                    if connectivity_type == 1
                                                        [gpdc_result, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, NSamp);
                                                    elseif connectivity_type == 2
                                                        [~, gpdc_result, ~, ~, ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, NSamp);
                                                    elseif connectivity_type == 3
                                                        [~, ~, gpdc_result, ~, ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, NSamp);
                                                    elseif connectivity_type == 4
                                                        [~, ~, ~, gpdc_result, ~, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, NSamp);
                                                    elseif connectivity_type == 5
                                                        [~, ~, ~, ~, gpdc_result, ~, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, NSamp);
                                                    elseif connectivity_type == 6
                                                        [~, ~, ~, ~, ~, gpdc_result, ~, ~, ~, ~, ~] = fdMVAR(eAm, eSu, nfft, NSamp);
                                                    end
                                                    
                                                    % Store GPDC result
                                                    GPDC{block}{cond, phrase}(:, :, :, w) = gpdc_result;
                                                    
                                                catch ME
                                                    % Handle MVAR estimation failures
                                                    GPDC{block}{cond, phrase}(:, :, :, w) = NaN(length(include)*2, length(include)*2, nfft);
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                
                %% Average GPDC over windows
                
                avGPDC = cell(1, 1);
                
                for block = 1:size(GPDC, 1)
                    if ~isempty(GPDC{block})
                        for cond = 1:size(GPDC{block}, 1)
                            for phrase = 1:size(GPDC{block}, 2)
                                if ~isempty(GPDC{block}{cond, phrase}) && size(GPDC{block}{cond, phrase}, 4) > 1
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
                end
                
                %% Extract connectivity matrices for each participant
                
                % Initialize quadrant storage
                GPDC_II = cell(size(avGPDC{p}));  % Infant-to-Infant
                GPDC_AA = cell(size(avGPDC{p}));  % Adult-to-Adult
                GPDC_AI = cell(size(avGPDC{p}));  % Adult-to-Infant
                GPDC_IA = cell(size(avGPDC{p}));  % Infant-to-Adult
                
                for block = 1:size(avGPDC{p}, 1)
                    for cond = 1:size(avGPDC{p}, 2)
                        for phase = 1:size(avGPDC{p}, 3)
                            if ~isempty(avGPDC{p}{block, cond, phase}) && ...
                               ~all(isnan(avGPDC{p}{block, cond, phase}(:)))
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
                
                % Initialize final storage
                II = cell(size(avGPDC{p}, 1), size(avGPDC{p}, 2), length(bands));
                AA = cell(size(avGPDC{p}, 1), size(avGPDC{p}, 2), length(bands));
                AI = cell(size(avGPDC{p}, 1), size(avGPDC{p}, 2), length(bands));
                IA = cell(size(avGPDC{p}, 1), size(avGPDC{p}, 2), length(bands));
                
                for block = 1:size(avGPDC{p}, 1)
                    for cond = 1:size(avGPDC{p}, 2)
                        % Initialize arrays to accumulate data across phases
                        tmp1All = [];  % For II
                        tmp2All = [];  % For AA
                        tmp3All = [];  % For AI
                        tmp4All = [];  % For IA
                        count = 0;     % Counter for valid phases
                        
                        for phase = 1:size(avGPDC{p}, 3)
                            if ~isempty(avGPDC{p}{block, cond, phase}) && ...
                               ~all(isnan(avGPDC{p}{block, cond, phase}(:)))
                                
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
                clear GPDC_II GPDC_AA GPDC_AI GPDC_IA avGPDC GPDC
                
                %% Save surrogate connectivity results
                
                % Save connectivity matrices
                if strcmp(current_location, 'C')
                    save_filename = fullfile(output_dir, ['UK_', participant_list{p}, '_PDC.mat']);
                else
                    save_filename = fullfile(output_dir, ['SG_', participant_list{p}, '_PDC.mat']);
                end
                
                save(save_filename, 'II', 'AI', 'AA', 'IA');
                
                fprintf('Processed participant %s from location %s for surrogate %d\n', ...
                       participant_list{p}, current_location, surr_idx);
                
            catch ME
                fprintf('Error processing participant %s: %s\n', participant_list{p}, ME.message);
            end
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
% 2. Creates surrogate data using a window shuffling approach:
%    - Extracts valid data windows across all conditions and phrases
%    - Randomly reassigns windows across conditions to break true connectivity
%    - Preserves spectral characteristics of individual windows
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

fprintf('\nSurrogate connectivity analysis complete.\n');
