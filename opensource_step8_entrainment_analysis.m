%% Neural Entrainment Analysis
% NOTE: This code demonstrates the analytical methodology only. Due to data privacy requirements,
% all data paths, variable names, and values shown here are examples only.
% The actual analysis was performed on protected research data. Data are protected and are not 
% available due to data privacy regulations. Access to anonymized data collected can be requested 
% by contacting the corresponding author (Prof. Victoria Leong, VictoriaLeong@ntu.edu.sg) and 
% is subject to the establishment of a specific data sharing agreement between the applicant's 
% institution and the institutions of data collection.
%
% Purpose: Calculate and analyze entrainment between speech envelopes and neural oscillations
% in infant EEG data across different experimental conditions (gaze manipulation)
%
% This script consists of two main parts:
% Part 1: Compute entrainment measures between speech and EEG signals
% Part 2: Process and analyze entrainment data across participants and conditions

%% =================== PART 1: COMPUTE ENTRAINMENT ===================

clear all;
dbstop if error

% Set base path
base_path = '/path/to/data/';

% Process both research locations
for location_idx = 1:2
    % Set location-specific parameters
    if location_idx == 2
        sg = 1;  % Singapore
        data_dir = fullfile(base_path, 'Preprocessed_Data_sg/');
        list_ppts = dir(fullfile(data_dir, 'P1*/P1*_BABBLE_AR.mat'));
    else
        sg = 0;  % Cambridge
        data_dir = fullfile(base_path, 'Preprocessed_Data_camb/');
        list_ppts = dir(fullfile(data_dir, 'P1*/P1*_BABBLE_AR.mat'));
    end
    
    % Define paths for different data types
    results_path = fullfile(base_path, 'data_matfile/');
    wav_file_dir = fullfile(base_path, 'entrain/*/audio_files');
    syll_file_dir = fullfile(base_path, 'entrain/*/');
    babble_dir = fullfile(base_path, 'entrain/*/*_orders_for_*/');
    
    % Get list of participant IDs
    participant_ids = {list_ppts.name}';
    
    %% Process each participant
    for pt = 1:length(participant_ids)
        id = participant_ids{pt};
        participant_name = id(1:5);
        
        fprintf('Processing participant %d of %d, ID %s\n', pt, length(participant_ids), participant_name);
        
        % Load participant-specific files
        list_files = dir(fullfile(strcat(syll_file_dir, '/*', id(3:5)), '*_AR_onset.mat'));
        list_files = {list_files.name}';
        
        % Extract order ID
        order_id = extractAfter(list_files, 'P');
        order_id = extractBefore(order_id, 'S_BABBLE_AR_onset.mat');
        
        if contains(list_files, 'C_BABBLE')
            order_id = extractAfter(list_files, 'P1');
            order_id = extractBefore(order_id, 'C_BABBLE_AR_onset.mat');
        end
        
        % Load babble order information
        fid = fopen(fullfile(babble_dir, 'orders.txt'));
        babble_orders = textscan(fid, '%f%f', 'delimiter', '/t');
        fclose(fid);
        
        babble_ppts = babble_orders{1, 1};
        babble_ppts_orders = babble_orders{1, 2};
        
        % Determine current order
        current_order = strcat('order', num2str(babble_ppts_orders(babble_ppts == str2num(order_id{1, 1}))));
        
        % Load stimulus IDs and onset times
        load(fullfile(babble_dir, 'wavs_in_order_correctedfororder23.mat'));
        stim_ids_all = wavs_in_order.(current_order);
        
        load(fullfile(babble_dir, 'downsampled_onset.mat'), 'ds_onset');
        stim_onsettimes_all = ds_onset{babble_ppts_orders(babble_ppts == str2num(order_id{1, 1})), 1};
        
        clear babble_orders babble_ppts babble_ppts_orders ds_onset wavs_in_order
        
        %% Set analysis parameters
        
        % Signal processing parameters
        fs = 200;  % Sampling rate (Hz)
        
        % Frequency bands for entrainment analysis
        delta_band = [1, 3];   % Delta: 1-3 Hz
        theta_band = [3, 6];   % Theta: 3-6 Hz
        alpha_band = [6, 9];   % Alpha: 6-9 Hz
        
        %% Define gaze condition order based on participant's experimental order
        
        if strcmp(current_order, 'order1')
            gaze_order_for_wav = [1, 2, 3];  % Full, Partial, No gaze
        elseif strcmp(current_order, 'order2')
            gaze_order_for_wav = [3, 1, 2];  % No, Full, Partial gaze
        elseif strcmp(current_order, 'order3')
            gaze_order_for_wav = [2, 3, 1];  % Partial, No, Full gaze
        end
        
        %% Process each session, block, condition, and phrase
        
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
                        %% Load and process audio
                        
                        % Get audio file path for current condition/phrase
                        audio_file = fullfile(wav_file_dir, ...
                            [stim_ids_all{gaze_order_for_wav(condition), phrase}, '.wav']);
                        
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
                        
                        % Find syllable markers (first 6 syllables, approximately 2 words)
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
                        
                        %% Extract matching speech envelope segment
                        
                        current_wav_onset = stim_onsettimes_all{condition, phrase};
                        onset_idx = find(current_wav_onset == 999, 1);
                        
                        if isempty(onset_idx)
                            continue;
                        end
                        
                        segment_length = syllable_markers(7) - syllable_markers(1);
                        audio_segment = audio_resampled(onset_idx:(onset_idx + segment_length - 1));
                        
                        %% Calculate cross-correlation between speech and neural envelopes
                        
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
        
        %% Save results for this participant
        
        filename = sprintf('BABBLE_%s_power_xcov_allbands_6syll_filtFULLSET_voltaverage_W9chans.mat', id);
        save(fullfile(results_path, filename), 'CondMatx', '-v7.3');
        
        % Clear large variables
        clear CondMatx syllable_markers
    end
end

% Note: The Hilb_Amp_BABBLE function is assumed to be available in the path
% This function computes the Hilbert transform in the specified frequency band


%% =================== PART 2: PROCESS AND ANALYZE ENTRAINMENT DATA ===================

clear all;
clc

% Set base path
base_path = '/path/to/data/';

% Define channel names for reference
channels = {'F3', 'FZ', 'F4', 'C3', 'CZ', 'C4', 'P3', 'PZ', 'P4'};

% Process both research locations
alpha_peak = [];
alpha_lag = [];
theta_peak = [];
theta_lag = [];
delta_peak = [];
delta_lag = [];

for location_idx = 1:2
    % Set location-specific parameters
    if location_idx == 2
        sg = 1;  % Singapore
        data_dir = fullfile(base_path, 'Preprocessed_Data_sg/');
        list_ppts = dir(fullfile(data_dir, 'P*/P1*_BABBLE_AR.mat'));
        
        % Define condition order for Singapore participants
        order_for_current_participants = [1, 1, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
    else
        sg = 0;  % Cambridge
        data_dir = fullfile(base_path, 'Preprocessed_Data_camb/');
        list_ppts = dir(fullfile(data_dir, 'P*/*_BABBLE_AR.mat'));
        
        % Define condition order for Cambridge participants
        order_for_current_participants = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 2];
    end
    
    results_path = fullfile(base_path, 'data_matfile/entrain');
    participant_ids = {list_ppts.name}';
    
    %% Load and extract entrainment data for each participant
    
    for pt = 1:length(participant_ids)
        % Load entrainment data
        filename = sprintf('BABBLE_%s_power_xcov_allbands_6syll_filtFULLSET_voltaverage_W9chans.mat', participant_ids{pt});
        load(fullfile(results_path, filename), 'CondMatx');
        
        % Define condition order (typically 1=Full gaze, 2=Partial gaze, 3=No gaze)
        conditions_order = [1, 2, 3];
        
        % Extract entrainment measures across sessions, blocks, conditions, and phrases
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
                                alpha_peak(:, conditions_order(condition), phrase, block, session, pt) = NaN;
                                alpha_lag(:, conditions_order(condition), phrase, block, session, pt) = NaN;
                                theta_peak(:, conditions_order(condition), phrase, block, session, pt) = NaN;
                                theta_lag(:, conditions_order(condition), phrase, block, session, pt) = NaN;
                                delta_peak(:, conditions_order(condition), phrase, block, session, pt) = NaN;
                                delta_lag(:, conditions_order(condition), phrase, block, session, pt) = NaN;
                            end
                        end
                    end
                end
            end
        end
    end
    
    %% Create table with all entrainment data
    
    % Initialize storage
    data = [];
    ids = {};
    
    % Process each participant
    for pt = 1:length(participant_ids)
        % Load entrainment data
        filename = sprintf('BABBLE_%s_power_xcov_allbands_6syll_filtFULLSET_voltaverage_W9chans.mat', participant_ids{pt});
        load(fullfile(results_path, filename), 'CondMatx');
        
        % Process each session, block, condition, and phrase
        for session = 1:1
            for block = 1:size(CondMatx, 2)
                for condition = 1:3
                    for phrase = 1:3
                        % Create row with all metrics
                        new_row = [
                            block, condition, phrase, ...
                            alpha_peak(:, condition, phrase, block, session, pt)', ...
                            alpha_lag(:, condition, phrase, block, session, pt)', ...
                            theta_peak(:, condition, phrase, block, session, pt)', ...
                            theta_lag(:, condition, phrase, block, session, pt)', ...
                            delta_peak(:, condition, phrase, block, session, pt)', ...
                            delta_lag(:, condition, phrase, block, session, pt)'
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
    
    %% Create table with descriptive column names
    
    % Define column names
    column_names = {'block', 'condition', 'phrase'};
    metrics = {'alpha_peak', 'alpha_lag', 'theta_peak', 'theta_lag', 'delta_peak', 'delta_lag'};
    
    for i = 1:length(metrics)
        for j = 1:length(channels)
            column_names = [column_names, sprintf('%s_%s', metrics{i}, channels{j})];
        end
    end
    
    % Create table
    result_table = array2table(data, 'VariableNames', column_names);
    
    % Add participant IDs
    result_table = addvars(result_table, ids, 'Before', 'block', 'NewVariableNames', {'ID'});
    
    % Ensure ID contains only the first 5 characters
    result_table.ID = cellfun(@(x) x(1:min(5, length(x))), result_table.ID, 'UniformOutput', false);
    
    % Display first few rows
    head(result_table);
    
    % Store table by location
    if sg == 1
        result_table_sg = result_table;
    else
        result_table_ca = result_table;
    end
end

%% Combine tables from both locations

table_combined = [result_table_ca; result_table_sg];

% Save combined table
save(fullfile(base_path, 'entrainment_table.mat'), 'table_combined');
writetable(table_combined, fullfile(base_path, 'entrainment_table.xlsx'));