%% Surrogate Neural Entrainment Analysis
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.
%
% Purpose: Generate surrogate data for neural entrainment analysis by randomizing 
% the relationship between speech and EEG signals, then compile surrogate data
% for statistical analysis
%
% This script consists of two main parts:
% Part 1: Generate surrogate data by randomizing speech-EEG pairings
% Part 2: Compile surrogate data for statistical analysis

%% =================== PART 1: GENERATE SURROGATE DATA ===================

clear all;
dbstop if error

% Set base path
base_path = '/path/to/data/';

% Number of surrogate iterations to generate
num_surrogates = 200;

% Process surrogate iterations
for surr_idx = 1:num_surrogates
    % Create output directory for this surrogate
    output_dir = fullfile(base_path, 'data_matfile/permuentrain', ['permu', num2str(surr_idx)]);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Process location 1 (Cambridge)
    sg = 0;
    
    % Define data paths
    results_path = fullfile(base_path, 'entrain/');
    data_dir = fullfile(base_path, 'Preprocessed_Data_camb/');
    wav_file_dir = fullfile(base_path, 'entrain/*/audio_files');
    syll_file_dir = fullfile(base_path, 'entrain/*/');
    babble_dir = fullfile(base_path, 'entrain/*/*_orders_for_*/');
    
    % Find participants
    list_ppts = dir(fullfile(data_dir, 'P1*/P1*_*_AR.mat'));
    participant_ids = {list_ppts.name}';
    
    % Process each participant
    for pt = 1:length(participant_ids)
        % Get participant ID
        id = participant_ids{pt};
        participant_name = id(1:5);
        
        fprintf('Processing participant %d of %d, ID %s\n', pt, length(participant_ids), participant_name);
        
        % Load participant-specific files
        list_files = dir(fullfile(strcat(syll_file_dir, '/*', id(3:5)), '*_AR_onset.mat'));
        list_files = {list_files.name}';
        
        % Extract order ID
        order_id = extractAfter(list_files, 'P');
        order_id = extractBefore(order_id, 'S_*_AR_onset.mat');
        
        if contains(list_files, 'C_BABBLE')
            order_id = extractAfter(list_files, 'P1');
            order_id = extractBefore(order_id, 'C_*_AR_onset.mat');
        end
        
        % Load babble order information
        fid = fopen(fullfile(babble_dir, 'orders.txt'));
        babble_orders = textscan(fid, '%f%f', 'delimiter', '/t');
        fclose(fid);
        
        babble_ppts = babble_orders{1, 1};
        babble_ppts_orders = babble_orders{1, 2};
        
        % Determine current order
        current_order = strcat('order', num2str(babble_ppts_orders(babble_ppts == str2num(order_id{1, 1}))));
        
        % Load stimulus information
        load(fullfile(babble_dir, 'wavs_in_order_correctedfororder23.mat'));
        
        % Randomly select order for surrogate data
        possible_orders = {'order1', 'order2', 'order3'};
        randomized_order = possible_orders{randi(3)};
        stim_ids_all = wavs_in_order.(randomized_order);
        
        % Load onset times
        load(fullfile(babble_dir, 'downsampled_onset.mat'), 'ds_onset');
        stim_onsettimes_all = ds_onset{babble_ppts_orders(babble_ppts == str2num(order_id{1, 1})), 1};
        
        clear babble_orders babble_ppts babble_ppts_orders ds_onset wavs_in_order
        
        % Set analysis parameters
        fs = 200;  % Sampling rate
        
        % Frequency bands for entrainment analysis
        delta_band = [1, 3];   % Delta: 1-3 Hz
        theta_band = [3, 6];   % Theta: 3-6 Hz
        alpha_band = [6, 9];   % Alpha: 6-9 Hz
        
        % Define correspondence between order and gaze conditions
        if strcmp(current_order, 'order1')
            gaze_order_for_wav = [1, 2, 3];  % Full, Partial, No gaze
        elseif strcmp(current_order, 'order2')
            gaze_order_for_wav = [3, 1, 2];  % No, Full, Partial gaze
        elseif strcmp(current_order, 'order3')
            gaze_order_for_wav = [2, 3, 1];  % Partial, No, Full gaze
        end
        
        % Process each session
        for session = 1:length(list_files)
            % Load EEG data
            if sg == 1
                participant_name = ['P0', participant_name(3:5)];
                load(fullfile(strcat(syll_file_dir, participant_name), '/', list_files{session}), 'FamEEGart');
            else
                load(fullfile(strcat(syll_file_dir, participant_name), '/', list_files{session}), 'FamEEGart');
            end
            
            % Process each familiarization block
            for block = 1:size(FamEEGart, 1)
                if isempty(FamEEGart{block})
                    continue;
                end
                
                % Process each condition (Full/Partial/No gaze)
                for condition = 1:size(FamEEGart{block, 1}, 1)
                    % Process each phrase
                    for phrase = 1:size(FamEEGart{block, 1}, 2)
                        %% Load and randomize audio
                        
                        % Randomly select condition and phrase for surrogate analysis
                        random_condition = randi(3);
                        random_phrase = randi(3);
                        
                        % Load random audio file
                        audio_file = fullfile(wav_file_dir, ...
                            [stim_ids_all{random_condition, random_phrase}, '.wav']);
                        
                        % Load audio and compute Hilbert envelope
                        [audio, audio_fs] = audioread(audio_file);
                        audio_envelope = abs(hilbert(audio));
                        
                        % Resample to EEG sampling rate
                        audio_resampled = resample(audio_envelope, fs, audio_fs);
                        
                        %% Process EEG data
                        
                        % Get EEG data for current condition/phrase
                        eeg_data = FamEEGart{block, 1}{condition, phrase}';
                        
                        % Skip if insufficient data
                        if size(eeg_data, 2) < 200
                            continue;
                        end
                        
                        % Find syllable markers (first 6 syllables)
                        syllable_markers = find(eeg_data(33, :) == 999, 7);
                        if length(syllable_markers) < 7
                            continue;
                        end
                        
                        % Extract channels of interest (9 channels: frontal, central, parietal)
                        roi_channels = [4:6, 15:17, 26:28];  % F3,Fz,F4,C3,Cz,C4,P3,Pz,P4
                        data = eeg_data(roi_channels, :);
                        
                        % Handle artifact markers
                        data(data == 999) = NaN;  % Event markers
                        data(data == 888) = NaN;  % Manual rejection
                        data(data == 777) = NaN;  % Unattended
                        
                        % Replace NaNs with zeros for filtering
                        data(isnan(data)) = 0;
                        
                        %% Create EEG structure for analysis
                        
                        EEGx = struct();
                        EEGx.nbchan = size(data, 1);  % Number of channels
                        EEGx.pnts = size(data, 2);    % Number of time points
                        EEGx.trials = 1;              % Single trial
                        EEGx.srate = fs;              % Sampling rate
                        EEGx.event = [];
                        EEGx.data = data;             % EEG data
                        
                        % Extract data for first 6 syllables
                        syllable_data = data(:, syllable_markers(1):syllable_markers(7) - 1);
                        
                        %% Calculate Hilbert envelopes for each frequency band
                        
                        % Alpha band (6-9 Hz)
                        alpha_hilbert = Hilb_Amp_BABBLE(EEGx, alpha_band);
                        alpha_amplitude = squeeze(abs(alpha_hilbert(1, :, syllable_markers(1):syllable_markers(7) - 1)));
                        alpha_amplitude(syllable_data == 0) = NaN';
                        
                        % Theta band (3-6 Hz)
                        theta_hilbert = Hilb_Amp_BABBLE(EEGx, theta_band);
                        theta_amplitude = squeeze(abs(theta_hilbert(1, :, syllable_markers(1):syllable_markers(7) - 1)));
                        theta_amplitude(syllable_data == 0) = NaN';
                        
                        % Delta band (1-3 Hz)
                        delta_hilbert = Hilb_Amp_BABBLE(EEGx, delta_band);
                        delta_amplitude = squeeze(abs(delta_hilbert(1, :, syllable_markers(1):syllable_markers(7) - 1)));
                        delta_amplitude(syllable_data == 0) = NaN';
                        
                        %% Calculate appropriate lag values for cross-correlation
                        
                        nlags_alpha = round(EEGx.srate / alpha_band(2));
                        nlags_theta = round(EEGx.srate / theta_band(2));
                        nlags_delta = round(EEGx.srate / delta_band(2));
                        
                        %% Extract randomized speech envelope segment
                        
                        % Calculate original length
                        original_length = syllable_markers(7) - syllable_markers(1);
                        
                        % Calculate maximum possible starting point
                        max_start = length(audio_resampled) - original_length + 1;
                        
                        % If audio is too short, skip
                        if max_start < 1
                            continue;
                        end
                        
                        % Pick a random starting point
                        random_start = randi(max_start);
                        
                        % Extract random segment matching original length
                        audio_segment = audio_resampled(random_start:(random_start + original_length - 1));
                        
                        %% Calculate cross-correlation between randomized speech and neural envelopes
                        
                        for channel = 1:EEGx.nbchan
                            % Alpha band
                            [xcov_alpha, lags_alpha] = xcov(tiedrank(audio_segment), ...
                                tiedrank(alpha_amplitude(channel, :)'), ...
                                nlags_alpha, 'coeff');
                            
                            % Convert lags to milliseconds
                            lags_alpha_ms = (lags_alpha * 1000) / fs;
                            
                            % Find peak correlation and its lag
                            [peak_alpha, lag_idx_alpha] = max(abs(xcov_alpha));
                            
                            % Store results
                            CondMatx(session, block).xcov_alpha{condition, phrase}(channel, :) = xcov_alpha;
                            
                            if ~isnan(peak_alpha)
                                CondMatx(session, block).alpha_peak{condition, phrase}(channel) = peak_alpha;
                                CondMatx(session, block).alpha_lag{condition, phrase}(channel) = lags_alpha_ms(lag_idx_alpha);
                            else
                                CondMatx(session, block).alpha_peak{condition, phrase}(channel) = NaN;
                                CondMatx(session, block).alpha_lag{condition, phrase}(channel) = NaN;
                            end
                            
                            % Theta band
                            [xcov_theta, lags_theta] = xcov(tiedrank(audio_segment), ...
                                tiedrank(theta_amplitude(channel, :)'), ...
                                nlags_theta, 'coeff');
                            
                            % Convert lags to milliseconds
                            lags_theta_ms = (lags_theta * 1000) / fs;
                            
                            % Find peak correlation and its lag
                            [peak_theta, lag_idx_theta] = max(abs(xcov_theta));
                            
                            % Store results
                            CondMatx(session, block).xcov_theta{condition, phrase}(channel, :) = xcov_theta;
                            
                            if ~isnan(peak_theta)
                                CondMatx(session, block).theta_peak{condition, phrase}(channel) = peak_theta;
                                CondMatx(session, block).theta_lag{condition, phrase}(channel) = lags_theta_ms(lag_idx_theta);
                            else
                                CondMatx(session, block).theta_peak{condition, phrase}(channel) = NaN;
                                CondMatx(session, block).theta_lag{condition, phrase}(channel) = NaN;
                            end
                            
                            % Delta band
                            [xcov_delta, lags_delta] = xcov(tiedrank(audio_segment), ...
                                tiedrank(delta_amplitude(channel, :)'), ...
                                nlags_delta, 'coeff');
                            
                            % Convert lags to milliseconds
                            lags_delta_ms = (lags_delta * 1000) / fs;
                            
                            % Find peak correlation and its lag
                            [peak_delta, lag_idx_delta] = max(abs(xcov_delta));
                            
                            % Store results
                            CondMatx(session, block).xcov_delta{condition, phrase}(channel, :) = xcov_delta;
                            
                            if ~isnan(peak_delta)
                                CondMatx(session, block).delta_peak{condition, phrase}(channel) = peak_delta;
                                CondMatx(session, block).delta_lag{condition, phrase}(channel) = lags_delta_ms(lag_idx_delta);
                            else
                                CondMatx(session, block).delta_peak{condition, phrase}(channel) = NaN;
                                CondMatx(session, block).delta_lag{condition, phrase}(channel) = NaN;
                            end
                        end
                        
                        %% Store data for reference
                        
                        CondMatx(session, block).EEG = EEGx;
                        CondMatx(session, block).Speech = audio_segment;
                        CondMatx(session, block).hilb_alpha = alpha_amplitude;
                        CondMatx(session, block).hilb_theta = theta_amplitude;
                        CondMatx(session, block).hilb_delta = delta_amplitude;
                    end
                end
            end
        end
        
        %% Save surrogate results for this participant
        
        filename = sprintf('%s_permu%d.mat', participant_name, surr_idx);
        save(fullfile(output_dir, filename), 'CondMatx', '-v7.3');
        
        % Clear large variables
        clear CondMatx syllable_markers
    end
    
    % Process location 2 (Singapore)
    sg = 1;
    data_dir = fullfile(base_path, 'Preprocessed_Data_sg/');
    
    % Find participants
    list_ppts = dir(fullfile(data_dir, 'P1*/P1*_BABBLE_AR.mat'));
    participant_ids = {list_ppts.name}';
    
    % Process each participant (code identical to above, with sg=1)
    % ... (Same processing as for Cambridge participants)
    
    fprintf('Completed surrogate iteration %d of %d\n', surr_idx, num_surrogates);
end


%% =================== PART 2: COMPILE SURROGATE DATA ===================

clear all;
clc

% Set base path
base_path = '/path/to/data/';
permutable = cell(1000, 1);  % Storage for surrogate datasets

% Number of surrogate iterations to process
num_surrogates = 1000;

% Process each surrogate iteration
for surr_idx = 1:num_surrogates
    fprintf('Processing surrogate %d of %d\n', surr_idx, num_surrogates);
    
    % Process both locations
    for location_idx = 1:2
        % Set location-specific parameters
        if location_idx == 2
            sg = 1;  % Singapore
            data_dir = fullfile(base_path, 'Preprocessed_Data_sg/');
        else
            sg = 0;  % Cambridge
            data_dir = fullfile(base_path, 'Preprocessed_Data_camb/');
        end
        
        % Define surrogate data path
        surr_path = fullfile(base_path, 'data_matfile/permuentrain', ['permu', num2str(surr_idx)]);
        
        % Find participants
        if sg == 1
            list_ppts = dir(fullfile(data_dir, 'P*/P1*_BABBLE_AR.mat'));
            order_for_current_participants = [1,1,3,1,2,1,3,2,3,1,2,3,1,2,3,1,2,3];
        else
            list_ppts = dir(fullfile(data_dir, 'P*/*_BABBLE_AR.mat'));
            order_for_current_participants = [1,2,3,1,2,3,1,2,3,2,3,2,1,2,3,1,3,1,2,3,1,2,3,1,2,1,2,3,2];
        end
        
        participant_ids = {list_ppts.name}';
        
        % Initialize data arrays
        alpha_peak = [];
        alpha_lag = [];
        theta_peak = [];
        theta_lag = [];
        delta_peak = [];
        delta_lag = [];
        
        % Process each participant
        for pt = 1:length(participant_ids)
            % Get participant name
            p_name = participant_ids{pt};
            if sg == 1
                p_name = ['P0', p_name(3:5)];
            else
                p_name = p_name(1:5);
            end
            
            % Load surrogate data for this participant
            filename = fullfile(surr_path, [p_name, '_permu', num2str(surr_idx), '.mat']);
            if ~exist(filename, 'file')
                continue;
            end
            
            load(filename, 'CondMatx');
            
            % Define condition order
            conditions_order = [1, 2, 3];
            
            % Extract entrainment measures
            for session = 1:1  % Typically just one session
                for block = 1:size(CondMatx, 2)
                    if ~isempty(CondMatx(session, block).alpha_peak)
                        for condition = 1:size(CondMatx(session, block).alpha_peak, 1)
                            for phrase = 1:size(CondMatx(session, block).alpha_peak, 2)
                                if ~isempty(CondMatx(session, block).alpha_peak{condition, phrase})
                                    % Store peak correlation values
                                    alpha_peak(:, conditions_order(condition), phrase, block, session, pt) = ...
                                        CondMatx(session, block).alpha_peak{condition, phrase};
                                    theta_peak(:, conditions_order(condition), phrase, block, session, pt) = ...
                                        CondMatx(session, block).theta_peak{condition, phrase};
                                    delta_peak(:, conditions_order(condition), phrase, block, session, pt) = ...
                                        CondMatx(session, block).delta_peak{condition, phrase};
                                    
                                    % Store lag values (in ms)
                                    alpha_lag(:, conditions_order(condition), phrase, block, session, pt) = ...
                                        CondMatx(session, block).alpha_lag{condition, phrase};
                                    theta_lag(:, conditions_order(condition), phrase, block, session, pt) = ...
                                        CondMatx(session, block).theta_lag{condition, phrase};
                                    delta_lag(:, conditions_order(condition), phrase, block, session, pt) = ...
                                        CondMatx(session, block).delta_lag{condition, phrase};
                                else
                                    % Fill with NaNs if data is missing
                                    alpha_peak(:, conditions_order(condition), phrase, block, session, pt) = NaN(1, 9);
                                    alpha_lag(:, conditions_order(condition), phrase, block, session, pt) = NaN(1, 9);
                                    theta_peak(:, conditions_order(condition), phrase, block, session, pt) = NaN(1, 9);
                                    theta_lag(:, conditions_order(condition), phrase, block, session, pt) = NaN(1, 9);
                                    delta_peak(:, conditions_order(condition), phrase, block, session, pt) = NaN(1, 9);
                                    delta_lag(:, conditions_order(condition), phrase, block, session, pt) = NaN(1, 9);
                                end
                            end
                        end
                    end
                end
            end
        end
        
        % Define channel names
        channels = {'F3', 'FZ', 'F4', 'C3', 'CZ', 'C4', 'P3', 'PZ', 'P4'};
        
        % Initialize data arrays for table
        data = [];
        ids = {};
        
        % Process each participant again to create table rows
        for pt = 1:length(participant_ids)
            p_name = participant_ids{pt};
            if sg == 1
                p_name = ['P0', p_name(3:5)];
            else
                p_name = p_name(1:5);
            end
            
            filename = fullfile(surr_path, [p_name, '_permu', num2str(surr_idx), '.mat']);
            if ~exist(filename, 'file')
                continue;
            end
            
            load(filename, 'CondMatx');
            
            % Process each session, block, condition, and phrase
            for session = 1:1
                for block = 1:size(CondMatx, 2)
                    for condition = 1:3
                        for phrase = 1:3
                            % Skip if data doesn't exist
                            if size(alpha_peak, 6) < pt || ...
                               size(alpha_peak, 1) < 1 || ...
                               isempty(squeeze(alpha_peak(:, condition, phrase, block, session, pt)))
                                continue;
                            end
                            
                            % Create row with all metrics
                            new_row = [
                                block, condition, phrase, ...
                                squeeze(alpha_peak(:, condition, phrase, block, session, pt))', ...
                                squeeze(alpha_lag(:, condition, phrase, block, session, pt))', ...
                                squeeze(theta_peak(:, condition, phrase, block, session, pt))', ...
                                squeeze(theta_lag(:, condition, phrase, block, session, pt))', ...
                                squeeze(delta_peak(:, condition, phrase, block, session, pt))', ...
                                squeeze(delta_lag(:, condition, phrase, block, session, pt))'
                            ];
                            
                            % Handle zeros (convert to NaN)
                            if length(find(new_row == 0)) < 5
                                data = [data; new_row];
                                ids = [ids; participant_ids(pt)];
                            else
                                new_row(new_row == 0) = NaN;
                                data = [data; new_row];
                                ids = [ids; participant_ids(pt)];
                            end
                        end
                    end
                end
            end
        end
        
        % Define column names
        column_names = {'block', 'condition', 'phrase'};
        metrics = {'alpha_peak', 'alpha_lag', 'theta_peak', 'theta_lag', 'delta_peak', 'delta_lag'};
        
        for i = 1:length(metrics)
            for j = 1:length(channels)
                column_names = [column_names, sprintf('%s_%s', metrics{i}, channels{j})];
            end
        end
        
        % Create table
        if isempty(data)
            continue;
        end
        
        result_table = array2table(data, 'VariableNames', column_names);
        
        % Store table by location
        if sg == 1
            result_table_sg = result_table;
            id_sg = ids;
        else
            result_table_ca = result_table;
            id_ca = ids;
        end
    end
    
    % Combine tables from both locations
    if exist('result_table_ca', 'var') && exist('result_table_sg', 'var')
        combined_table = [result_table_ca; result_table_sg];
        combined_ids = [id_ca; id_sg];
        combined_table = addvars(combined_table, combined_ids, 'Before', 'block', 'NewVariableNames', {'ID'});
        
        % Store in surrogate array
        permutable{surr_idx} = combined_table;
    end
end

% Remove empty surrogates
empty_idx = cellfun(@isempty, permutable);
permutable(empty_idx) = [];

% Save surrogate data
save(fullfile(base_path, 'ENTRIANSURR.mat'), 'permutable', '-v7.3');

fprintf('Surrogate data compilation complete.\n');

%% =================== HELPER FUNCTIONS ===================

function hilb_out = Hilb_Amp_BABBLE(EEG, freq_range)
    % Initialize output array
    hilb_out = zeros(1, EEG.nbchan, EEG.pnts);
    
    % Design bandpass filter for the specified frequency range
    [b, a] = butter(3, [freq_range(1) freq_range(2)]/(EEG.srate/2), 'bandpass');
    
    % Apply filter and Hilbert transform to each channel
    for chan = 1:EEG.nbchan
        % Apply bandpass filter
        filtered_data = filtfilt(b, a, double(EEG.data(chan, :)));
        
        % Apply Hilbert transform to get analytic signal
        hilb_out(1, chan, :) = hilbert(filtered_data);
    end
end