%% EEG Attendance Analysis Script
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Calculate attendance onset and duration based on the segments in infant EEG data
%
% This script:
% 1. Loads preprocessed EEG data with attendance markers
% 2. Identifies segments of continuous attention
% 3. Calculates onset points, durations, and number of attention segments
% 4. Verifies data integrity by comparing with filtered EEG data
%
% Attendance codes: 
% - 777 = unattended
% - 888 = manually rejected
% - 999 = automatically rejected
% - NaN = missing data

%% Setup parameters and paths

% Clear workspace
clear all
clc

% Set base path (modify as needed)
base_path = '/path/to/data/';

% Analysis parameters
location = 'C';  % C = location 1, S = location 2
montage = 'GRID';

% Define frequency bins for spectral analysis
freqs = [9, 17, 24, 32]; % Frequency bins corresponding to 3, 6.25, 9, 12.1 Hz for nfft = 256

% Define file paths based on location
if strcmp(location, 'S')
    filepath = fullfile(base_path, 'Preprocessed_Data_sg/');
else
    filepath = fullfile(base_path, 'Preprocessed_Data_location1/');
end

% Define electrode montage
if strcmp(montage, 'AP3')
    include = [5, 16, 27]'; % Anterior-Posterior Midline
elseif strcmp(montage, 'AP4')
    include = [4, 6, 26, 28]'; % Anterior-Posterior Left-Right (square)
elseif strcmp(montage, 'CM3')
    include = [15:17]'; % Central Motor Midline
elseif strcmp(montage, 'APM4')
    include = [15, 17, 5, 27]'; % Anterior-Posterior Midline (diamond)
elseif strcmp(montage, 'FA3')
    include = [4:6]'; % Frontal Left-Right
elseif strcmp(montage, 'PR3')
    include = [26, 27, 28]'; % Parietal [P3, Pz, P4]
elseif strcmp(montage, 'GRID')
    include = [4:6, 15:17, 26:28]'; % 9-channel grid FA3, CM3, PR3
    chLabel = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
end

% Signal processing parameters
NSamp = 200; % Sampling rate (Hz)
len = 1.5; % Window length in seconds (default 1.5s)
shift = 0.5 * len * NSamp; % Overlap between windows (default 0.5)
wlen = len * NSamp; % Window length in samples
nfft = 256; % FFT size for spectral analysis
nshuff = 0; % Number of shuffles (0 for no shuffling)

%% Load participant lists based on location

if strcmp(location, 'S')
    % Singapore participants (excluding problematic IDs)
    participant_list = {'101', '104', '106', '107', '108', '110', '112', '114', '115', ...
                        '116', '117', '118', '119', '120', '121', '122', '123', '127'};
else
    % Location 1 participants
    participant_list = {'101', '102', '103', '104', '105', '106', '107', '108', '109', ...
                        '111', '112', '114', '116', '117', '118', '119', '121', '122', ...
                        '123', '124', '125', '126', '127', '128', '129', '131', '132', ...
                        '133', '135'};
end

%% Initialize data structures for storing results

number_of_cond = 3;  % Number of conditions (Full, Partial, No gaze)
number_of_block = 3; % Number of experimental blocks
number_of_phrase = 3; % Number of phrases per condition

% Create cell arrays to store attendance data
onset_startpoint = cell(length(participant_list), number_of_cond, number_of_block, number_of_phrase);
onset_endpoint = cell(length(participant_list), number_of_cond, number_of_block, number_of_phrase);
onset_number = cell(length(participant_list), number_of_cond, number_of_block, number_of_phrase);
onset_duration = cell(length(participant_list), number_of_cond, number_of_block, number_of_phrase);

%% Process each participant

fprintf('Processing %d participants...\n', length(participant_list));

for p = 1:length(participant_list)
    fprintf('Processing participant %s (%d/%d)\n', participant_list{p}, p, length(participant_list));
    
    try
        % Initialize EEG data structures
        iEEG = [];
        aEEG = [];
        
        % Load attendance-marked EEG data
        filename = fullfile(filepath, ['P', participant_list{p}, location], ...
                           ['P', participant_list{p}, location, '_BABBLE_attend.mat']);
        
        if exist(filename, 'file')
            load(filename, 'FamEEGattend', 'StimEEGattend');
        else
            fprintf('Warning: File not found for participant %s\n', participant_list{p});
            continue;
        end
        
        % Extract selected channels and mark bad sections
        for block = 1:size(FamEEGattend, 1)
            if ~isempty(FamEEGattend{block})
                for cond = 1:size(FamEEGattend{block}, 1)
                    for phrase = 1:size(FamEEGattend{block}, 2)
                        if ~isempty(FamEEGattend{block}{cond, phrase}) && ...
                           size(FamEEGattend{block}{cond, phrase}, 1) > 1
                            
                            for chan = 1:length(include)
                                % Extract selected channels
                                iEEG{block}{cond, phrase}(:, chan) = ...
                                    FamEEGattend{block}{cond, phrase}(:, include(chan));
                                aEEG{block}{cond, phrase}(:, chan) = ...
                                    StimEEGattend{block}{cond, phrase}(:, include(chan));
                                
                                % Mark rejected/unattended segments with value 1000
                                rejected_idx = find(FamEEGattend{block}{cond, phrase}(:, chan) == 777 | ...
                                                  FamEEGattend{block}{cond, phrase}(:, chan) == 888 | ...
                                                  FamEEGattend{block}{cond, phrase}(:, chan) == 999);
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
        
        % Combine adult and infant EEG data horizontally
        for block = 1:size(FamEEGattend, 1)
            for cond = 1:size(FamEEGattend{block}, 1)
                for phrase = 1:size(FamEEGattend{block}, 2)
                    if ~isempty(aEEG{block}) && ~isempty(iEEG{block})
                        EEG{block, 1}{cond, phrase} = horzcat(aEEG{block}{cond, phrase}, ...
                                                             iEEG{block}{cond, phrase});
                    end
                end
            end
        end
        
        % Clear intermediate variables to free memory
        clear aEEG iEEG FamEEGattend StimEEGattend
        
        % Load filtered EEG data for verification
        filename2 = fullfile(filepath, ['P', participant_list{p}, location], ...
                            ['P', participant_list{p}, location, '_BABBLE_PP.mat']);
        
        if exist(filename2, 'file')
            load(filename2, 'FamEEGfil', 'StimEEGfil');
            
            % Process filtered data similar to attendance data
            for block = 1:size(FamEEGfil, 1)
                if ~isempty(FamEEGfil{block})
                    for cond = 1:size(FamEEGfil{block}, 1)
                        for phrase = 1:size(FamEEGfil{block}, 2)
                            if ~isempty(FamEEGfil{block}{cond, phrase}) && ...
                               size(FamEEGfil{block}{cond, phrase}, 1) > 1
                                
                                for chan = 1:length(include)
                                    iEEG{block}{cond, phrase}(:, chan) = ...
                                        FamEEGfil{block}{cond, phrase}(:, include(chan));
                                    aEEG{block}{cond, phrase}(:, chan) = ...
                                        StimEEGfil{block}{cond, phrase}(:, include(chan));
                                    
                                    % Mark rejected segments
                                    rejected_idx = find(FamEEGfil{block}{cond, phrase}(:, chan) == 777 | ...
                                                      FamEEGfil{block}{cond, phrase}(:, chan) == 888 | ...
                                                      FamEEGfil{block}{cond, phrase}(:, chan) == 999);
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
            
            % Combine filtered data
            for block = 1:size(FamEEGfil, 1)
                for cond = 1:size(FamEEGfil{block}, 1)
                    for phrase = 1:size(FamEEGfil{block}, 2)
                        if ~isempty(aEEG{block}) && ~isempty(iEEG{block})
                            EEG2{block, 1}{cond, phrase} = horzcat(aEEG{block}{cond, phrase}, ...
                                                                  iEEG{block}{cond, phrase});
                        end
                    end
                end
            end
            
            clear aEEG iEEG FamEEGfil StimEEGfil
        end
        
        % Calculate onset and duration of attended segments
        for block = 1:size(EEG, 1)
            if ~isempty(EEG{block})
                for cond = 1:size(EEG{block}, 1)
                    for phrase = 1:size(EEG{block}, 2)
                        if ~isempty(EEG{block}{cond, phrase}) && size(EEG{block}{cond, phrase}, 1) > 1
                            
                            % Find any non-NaN values across channels
                            [rows, cols] = find(~isnan(EEG{block}{cond, phrase}));
                            
                            if ~isempty(cols)
                                % Take the first channel with non-NaN values
                                temp = EEG{block}{cond, phrase}(:, cols(1));
                                nan_idx = isnan(temp);
                                
                                % Find start points of segments (NaN to non-NaN transitions)
                                segment_start_idx = find(diff([1; nan_idx]) == -1);
                                
                                % Find end points of segments (non-NaN to NaN transitions)
                                segment_end_idx = find(diff([nan_idx; 1]) == 1);
                                
                                % Handle edge cases
                                if ~isnan(EEG{block}{cond, phrase}(1, cols(1)))
                                    segment_start_idx = [1; segment_start_idx];
                                end
                                
                                if ~isnan(EEG{block}{cond, phrase}(end, cols(1)))
                                    segment_end_idx = [segment_end_idx; size(EEG{block}{cond, phrase}, 1)];
                                end
                                
                                % Calculate segment lengths
                                if length(segment_start_idx) == length(segment_end_idx)
                                    segment_length = segment_end_idx - segment_start_idx + 1;
                                else
                                    segment_length = [];
                                end
                            else
                                % No non-NaN data found
                                segment_start_idx = [];
                                segment_end_idx = [];
                                segment_length = [];
                            end
                            
                            % Verification check with filtered EEG data if available
                            verification_passed = true;
                            if exist('EEG2', 'var') && ~isempty(EEG2{block}{cond, phrase})
                                [rows2, cols2] = find(~isnan(EEG2{block}{cond, phrase}));
                                
                                if ~isempty(cols2)
                                    temp2 = EEG2{block}{cond, phrase}(:, cols2(1));
                                    nan_idx2 = isnan(temp2);
                                    segment_start_idx2 = find(diff([1; nan_idx2]) == -1);
                                    
                                    if ~isnan(EEG2{block}{cond, phrase}(1, cols2(1)))
                                        segment_start_idx2 = [1; segment_start_idx2];
                                    end
                                    
                                    % Simple verification: check if first segments align
                                    if ~isempty(segment_start_idx) && ~isempty(segment_start_idx2)
                                        verification_passed = (segment_start_idx(1) == segment_start_idx2(1));
                                    end
                                end
                            end
                            
                            % Store results if verification passes
                            if verification_passed
                                onset_startpoint{p, cond, block, phrase} = segment_start_idx;
                                onset_endpoint{p, cond, block, phrase} = segment_end_idx;
                                onset_number{p, cond, block, phrase} = length(segment_start_idx);
                                onset_duration{p, cond, block, phrase} = segment_length;
                            else
                                fprintf('Verification failed for P%s B%d C%d Ph%d\n', ...
                                       participant_list{p}, block, cond, phrase);
                                onset_startpoint{p, cond, block, phrase} = [];
                                onset_endpoint{p, cond, block, phrase} = [];
                                onset_number{p, cond, block, phrase} = NaN;
                                onset_duration{p, cond, block, phrase} = [];
                            end
                        else
                            % No data for this condition/phrase combination
                            onset_startpoint{p, cond, block, phrase} = [];
                            onset_endpoint{p, cond, block, phrase} = [];
                            onset_number{p, cond, block, phrase} = NaN;
                            onset_duration{p, cond, block, phrase} = [];
                        end
                    end
                end
            end
        end
        
        % Clear large EEG data to free memory
        clear EEG EEG2
        
    catch ME
        fprintf('Error processing participant %s: %s\n', participant_list{p}, ME.message);
        
        % Fill with empty/NaN values for this participant
        for cond = 1:number_of_cond
            for block = 1:number_of_block
                for phrase = 1:number_of_phrase
                    onset_startpoint{p, cond, block, phrase} = [];
                    onset_endpoint{p, cond, block, phrase} = [];
                    onset_number{p, cond, block, phrase} = NaN;
                    onset_duration{p, cond, block, phrase} = [];
                end
            end
        end
    end
    
    fprintf('Completed participant %d of %d\n', p, length(participant_list));
end

%% Save results

fprintf('Saving results...\n');

if strcmp(location, 'S')
    save(fullfile(base_path, 'attendance_data_location2.mat'), 'onset_startpoint', 'onset_endpoint', ...
         'onset_number', 'onset_duration', 'participant_list');
else
    save(fullfile(base_path, 'attendance_data_location1.mat'), 'onset_startpoint', 'onset_endpoint', ...
         'onset_number', 'onset_duration', 'participant_list');
end

%% Summary statistics

fprintf('\nCalculating summary statistics...\n');

% Initialize counters
total_segments = zeros(length(participant_list), 1);
all_durations_per_participant = cell(length(participant_list), 1);

for p = 1:length(participant_list)
    participant_durations = [];
    
    for cond = 1:number_of_cond
        for block = 1:number_of_block
            for phrase = 1:number_of_phrase
                if ~isempty(onset_number{p, cond, block, phrase}) && ~isnan(onset_number{p, cond, block, phrase})
                    total_segments(p) = total_segments(p) + onset_number{p, cond, block, phrase};
                end
                
                if ~isempty(onset_duration{p, cond, block, phrase})
                    participant_durations = [participant_durations; onset_duration{p, cond, block, phrase}];
                end
            end
        end
    end
    
    all_durations_per_participant{p} = participant_durations;
end

% Calculate statistics for participants with data
valid_participants = ~isnan(total_segments) & total_segments > 0;

if any(valid_participants)
    fprintf('Average number of attendance segments per participant: %.2f (SD = %.2f)\n', ...
            mean(total_segments(valid_participants)), std(total_segments(valid_participants)));
    
    % Calculate mean duration statistics
    mean_durations = zeros(length(participant_list), 1);
    median_durations = zeros(length(participant_list), 1);
    
    for p = 1:length(participant_list)
        if ~isempty(all_durations_per_participant{p})
            mean_durations(p) = mean(all_durations_per_participant{p});
            median_durations(p) = median(all_durations_per_participant{p});
        else
            mean_durations(p) = NaN;
            median_durations(p) = NaN;
        end
    end
    
    % Convert sample durations to time in seconds
    mean_durations_sec = mean_durations / NSamp;
    median_durations_sec = median_durations / NSamp;
    
    % Report statistics for participants with valid data
    valid_duration_participants = ~isnan(mean_durations_sec);
    
    if any(valid_duration_participants)
        fprintf('Average attendance segment duration: %.2f sec (SD = %.2f sec)\n', ...
                mean(mean_durations_sec(valid_duration_participants)), ...
                std(mean_durations_sec(valid_duration_participants)));
        fprintf('Median attendance segment duration: %.2f sec (SD = %.2f sec)\n', ...
                mean(median_durations_sec(valid_duration_participants)), ...
                std(median_durations_sec(valid_duration_participants)));
    end
    
    fprintf('Number of participants with valid attendance data: %d/%d\n', ...
            sum(valid_participants), length(participant_list));
else
    fprintf('Warning: No valid attendance data found for any participants.\n');
end

if strcmp(location, 'S')
    fprintf('\nProcessing complete. Results saved to attendance_data_location2.mat\n');
else
    fprintf('\nProcessing complete. Results saved to attendance_data_location1.mat\n');
end
