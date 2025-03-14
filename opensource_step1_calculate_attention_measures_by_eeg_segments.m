%% EEG Attendance Analysis Script
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.
%
% Purpose: Calculate attendance onset and duration based on the segments in infant EEG data
%
% This script:
% 1. Loads preprocessed EEG data with attendance markers
% 2. Identifies segments of continuous attention
% 3. Calculates onset points, durations, and number of attention segments
% 4. Verifies data integrity by comparing with raw EEG
%
% Attendance codes: 
% - 777 = unattended
% - 888 = manually rejected
% - 999 = automatically rejected
% - NaN = missing data

%% Setup parameters and paths

% Clear workspace
clear all

% Set base path (modify as needed)
base_path = '/path/to/data/';

% Analysis parameters
location = 'C';  % C = location 1, S = location 2
montage = 'GRID';

% Define file paths based on location
if strcmp(location, 'S')
    filepath = fullfile(base_path, 'dataset1/');
else
    filepath = fullfile(base_path, 'dataset2/');
end

% Define electrode montage
if strcmp(montage, 'GRID')
    include = [4:6, 15:17, 26:28]';  % 9-channel grid
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

% Recording parameters
fs = 200;  % Sampling rate (Hz)
epoch_length = 1.5;  % Length of epochs in seconds
window_length = epoch_length * fs;  % Window length in samples
window_overlap = 0.5;  % Overlap between windows (proportion)
fft_size = 256;  % FFT size for spectral analysis

%% Load participant lists based on location

if strcmp(location, 'S')
    participant_list = {'101', '104', '106', '107', '108', '110', '112', '114', '115', 
                         '116', '117', '118', '119', '120', '121', '122', '123', '127'};
else
    participant_list = {'101', '102', '103', '104', '105', '106', '107', '108', '109', 
                         '111', '112', '114', '116', '117', '118', '119', '121', '122',
                         '123', '124', '125', '126', '127', '128', '129', '131', '132', 
                         '133', '135'};
end

%% Initialize data structures for storing results

num_conditions = 3;
num_blocks = 3;
num_phrases = 3;

% Create cell arrays to store attendance data
onset_startpoint = cell(length(participant_list), num_conditions, num_blocks, num_phrases);
onset_endpoint = cell(length(participant_list), num_conditions, num_blocks, num_phrases);
onset_number = cell(length(participant_list), num_conditions, num_blocks, num_phrases);
onset_duration = cell(length(participant_list), num_conditions, num_blocks, num_phrases);

%% Process each participant

for p = 1:length(participant_list)
    % Initialize EEG data structures
    iEEG = [];
    aEEG = [];
    
    % Load attendance-marked EEG data
    filename = fullfile(filepath, ['P', participant_list{p}, location], ...
                        ['P', participant_list{p}, location, '*_attend.mat']);
    load(filename, 'FamEEGattend', 'StimEEGattend');
    
    % Extract selected channels and mark bad sections
    for block = 1:size(FamEEGattend, 1)
        if ~isempty(FamEEGattend{block})
            for cond = 1:size(FamEEGattend{block}, 1)
                for phrase = 1:size(FamEEGattend{block}, 2)
                    if ~isempty(FamEEGattend{block}{cond, phrase}) && size(FamEEGattend{block}{cond, phrase}, 1) > 1
                        for chan = 1:length(include)
                            % Extract selected channels
                            iEEG{block}{cond, phrase}(:, chan) = FamEEGattend{block}{cond, phrase}(:, include(chan));
                            aEEG{block}{cond, phrase}(:, chan) = StimEEGattend{block}{cond, phrase}(:, include(chan));
                            
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
    
    % Combine adult and infant EEG
    for block = 1:size(FamEEGattend, 1)
        for cond = 1:size(FamEEGattend{block}, 1)
            for phrase = 1:size(FamEEGattend{block}, 2)
                EEG{block, 1}{cond, phrase} = horzcat(aEEG{block}{cond, phrase}, iEEG{block}{cond, phrase});
            end
        end
    end
    
    % Clear intermediate variables to free memory
    clear aEEG iEEG FamEEGattend StimEEGattend
    
    % Load filtered EEG data for verification
    iEEG = [];
    aEEG = [];
    filename = fullfile(filepath, ['P', participant_list{p}, location], ...
                        ['P', participant_list{p}, location, '*_PP.mat']);
    load(filename, 'FamEEGfil', 'StimEEGfil');
    
    % Extract selected channels and mark bad sections in filtered data
    for block = 1:size(FamEEGfil, 1)
        if ~isempty(FamEEGfil{block})
            for cond = 1:size(FamEEGfil{block}, 1)
                for phrase = 1:size(FamEEGfil{block}, 2)
                    if ~isempty(FamEEGfil{block}{cond, phrase}) && size(FamEEGfil{block}{cond, phrase}, 1) > 1
                        for chan = 1:length(include)
                            % Extract selected channels
                            iEEG{block}{cond, phrase}(:, chan) = FamEEGfil{block}{cond, phrase}(:, include(chan));
                            aEEG{block}{cond, phrase}(:, chan) = StimEEGfil{block}{cond, phrase}(:, include(chan));
                            
                            % Mark rejected/unattended segments with value 1000
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
    
    % Combine adult and infant filtered EEG
    for block = 1:size(FamEEGfil, 1)
        for cond = 1:size(FamEEGfil{block}, 1)
            for phrase = 1:size(FamEEGfil{block}, 2)
                EEG2{block, 1}{cond, phrase} = horzcat(aEEG{block}{cond, phrase}, iEEG{block}{cond, phrase});
            end
        end
    end
    
    % Clear intermediate variables to free memory
    clear aEEG iEEG FamEEGfil StimEEGfil
    
    fprintf('Processing participant %s\n', participant_list{p});
    
    % Find and save the onset and duration of non-NaN segments (periods of attention)
    for block = 1:size(EEG, 1)
        if ~isempty(EEG{block})
            for cond = 1:size(EEG{block}, 1)
                for phrase = 1:size(EEG{block}, 2)
                    if ~isempty(EEG{block}{cond, phrase}) && size(EEG{block}{cond, phrase}, 1) > 1
                        % Find any non-NaN values
                        [rows, cols] = find(~isnan(EEG{block}{cond, phrase}));
                        
                        if ~isempty(cols)
                            % Take the first channel with non-NaN values
                            temp = EEG{block}{cond, phrase}(:, cols(1));
                            nan_idx = isnan(temp);
                            
                            % Find start points of segments (where data transitions from NaN to non-NaN)
                            segment_start_idx = find(diff([0; nan_idx]) == -1);
                            
                            % Find end points of segments (where data transitions from non-NaN to NaN)
                            segment_end_idx = find(diff([nan_idx; 0]) == 1);
                            
                            % If data begins with non-NaN, add 1 as the first start point
                            if ~isnan(EEG{block}{cond, phrase}(1, cols(1)))
                                segment_start_idx = [1; segment_start_idx];
                            end
                            
                            % If data ends with non-NaN, add end of data as the last end point
                            if ~isnan(EEG{block}{cond, phrase}(end, cols(1)))
                                segment_end_idx = [segment_end_idx; size(EEG{block}{cond, phrase}, 1)];
                            end
                            
                            % Calculate segment lengths
                            segment_length = segment_end_idx - segment_start_idx + 1;
                        else
                            % No non-NaN data found
                            segment_start_idx = 1;
                            segment_end_idx = [];
                            segment_length = [];
                        end
                        
                        % Verification check with filtered EEG data
                        [rows, cols] = find(~isnan(EEG2{block}{cond, phrase}));
                        
                        if ~isempty(cols)
                            temp = EEG2{block}{cond, phrase}(:, cols(1));
                            nan_idx = isnan(temp);
                            segment_start_idx2 = find(diff([0; nan_idx]) == -1);
                            
                            if ~isnan(EEG2{block}{cond, phrase}(1, cols(1)))
                                segment_start_idx2 = [1; segment_start_idx2];
                            end
                        else
                            segment_start_idx2 = 1;
                        end
                        
                        % If verification passes, store the results
                        if segment_start_idx2 == 1
                            onset_startpoint{p, cond, block, phrase} = segment_start_idx;
                            onset_endpoint{p, cond, block, phrase} = segment_end_idx;
                            onset_number{p, cond, block, phrase} = length(segment_start_idx);
                            onset_duration{p, cond, block, phrase} = segment_length;
                            
                            % Debugging plot (uncomment if needed)
                            % plot(EEG{block}{cond, phrase}(:, 4))
                        else
                            fprintf('Verification failed for P%s B%d C%d Ph%d\n', 
                                   participant_list{p}, block, cond, phrase);
                        end
                    else
                        % No data for this condition/phrase combination
                        onset_startpoint{p, cond, block, phrase} = [];
                        onset_endpoint{p, cond, block, phrase} = [];
                        onset_number{p, cond, block, phrase} = [];
                        onset_duration{p, cond, block, phrase} = [];
                    end
                end
            end
        end
    end
    
    % Clear large EEG data to free memory
    clear EEG EEG2
    fprintf('Completed participant %d of %d\n', p, length(participant_list));
end

%% Save results
if strcmp(location, 'S')
    save('attendance_data_location2.mat', 'onset_startpoint', 'onset_endpoint', 
         'onset_number', 'onset_duration', 'participant_list');
else
    save('attendance_data_location1.mat', 'onset_startpoint', 'onset_endpoint', 
         'onset_number', 'onset_duration', 'participant_list');
end

%% Example analysis (uncomment to use)
% Calculate average number of attendance segments per participant

% Initialize counters
% total_segments = zeros(length(participant_list), 1);
% 
% for p = 1:length(participant_list)
%     for cond = 1:num_conditions
%         for block = 1:num_blocks
%             for phrase = 1:num_phrases
%                 if ~isempty(onset_number{p, cond, block, phrase})
%                     total_segments(p) = total_segments(p) + onset_number{p, cond, block, phrase};
%                 end
%             end
%         end
%     end
% end
% 
% % Calculate mean and standard deviation
% fprintf('Average number of attendance segments per participant: %.2f (SD = %.2f)\n', 
%         mean(total_segments), std(total_segments));
% 
% % Example: Calculate mean duration of attendance segments
% all_durations = cell(length(participant_list), 1);
% 
% for p = 1:length(participant_list)
%     for cond = 1:num_conditions
%         for block = 1:num_blocks
%             for phrase = 1:num_phrases
%                 if ~isempty(onset_duration{p, cond, block, phrase})
%                     all_durations{p} = [all_durations{p}; onset_duration{p, cond, block, phrase}];
%                 end
%             end
%         end
%     end
% end
% 
% % Calculate statistics for each participant
% mean_durations = zeros(length(participant_list), 1);
% std_durations = zeros(length(participant_list), 1);
% median_durations = zeros(length(participant_list), 1);
% 
% for p = 1:length(participant_list)
%     if ~isempty(all_durations{p})
%         mean_durations(p) = mean(all_durations{p});
%         std_durations(p) = std(all_durations{p});
%         median_durations(p) = median(all_durations{p});
%     end
% end
% 
% % Convert sample durations to time in seconds
% mean_durations_sec = mean_durations / fs;
% median_durations_sec = median_durations / fs;
% 
% fprintf('Average attendance segment duration: %.2f sec (SD = %.2f sec)\n', 
%         mean(mean_durations_sec), std(mean_durations_sec));
% fprintf('Median attendance segment duration: %.2f sec\n', 
%         mean(median_durations_sec));