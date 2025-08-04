%% Neural Entrainment Analysis
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Calculate neural-speech entrainment (NSE) using cross-correlation analysis
% between infant EEG signals and the amplitude envelope of speech stimuli
%
% This script:
% 1. Loads preprocessed EEG data and speech audio files
% 2. Computes Hilbert transform envelopes for speech and EEG in delta, theta, alpha bands
% 3. Calculates cross-correlation between speech envelope and EEG power
% 4. Extracts peak correlation values and corresponding lags
% 5. Saves entrainment data for statistical analysis

%% =================== PART 1: CALCULATE ENTRAINMENT DATA ===================

clc;
clear all; 

% Set base paths
base_path = '/path/to/data/';
results_path = fullfile(base_path, 'entrainment_results');
audio_path = fullfile(base_path, 'audio_files');
eeg_path = fullfile(base_path, 'preprocessed_eeg');

% Create results directory if it doesn't exist
if ~exist(results_path, 'dir')
    mkdir(results_path);
end

%% Process data for both locations

for location = 1:2
    % Set location-specific parameters
    if location == 1
        location_name = 'UK';
        participant_files = dir(fullfile(eeg_path, 'P1*_BABBLE_AR.mat'));
    else
        location_name = 'SG';
        participant_files = dir(fullfile(eeg_path, 'P1*_BABBLE_AR.mat'));
    end
    
    % Extract participant IDs
    participant_ids = {participant_files.name}';
    participant_ids = cellfun(@(x) x(1:5), participant_ids, 'UniformOutput', false);
    
    fprintf('Processing %d participants from %s\n', length(participant_ids), location_name);
    
    %% Define analysis parameters
    
    % EEG sampling rate after preprocessing
    fs = 200;  % Hz
    
    % Frequency bands for entrainment analysis
    delta_band = [1, 3];   % Delta: 1-3 Hz
    theta_band = [3, 6];   % Theta: 3-6 Hz  
    alpha_band = [6, 9];   % Alpha: 6-9 Hz
    
    % EEG channels of interest (9 channels: F3, Fz, F4, C3, Cz, C4, P3, Pz, P4)
    roi_channels = [4:6, 15:17, 26:28];
    channels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
    
    %% Initialize storage arrays
    
    % Pre-allocate arrays for entrainment data
    % Dimensions: [channels, conditions, phrases, blocks, sessions, participants]
    alpha_peak = NaN(length(channels), 3, 3, 3, 1, length(participant_ids));
    alpha_lag = NaN(length(channels), 3, 3, 3, 1, length(participant_ids));
    theta_peak = NaN(length(channels), 3, 3, 3, 1, length(participant_ids));
    theta_lag = NaN(length(channels), 3, 3, 3, 1, length(participant_ids));
    delta_peak = NaN(length(channels), 3, 3, 3, 1, length(participant_ids));
    delta_lag = NaN(length(channels), 3, 3, 3, 1, length(participant_ids));
    
    %% Process each participant
    
    for pt = 1:length(participant_ids)
        id = participant_ids{pt};
        fprintf('Processing participant %d/%d: %s\n', pt, length(participant_ids), id);
        
        % Load experimental order information
        order_file = fullfile(base_path, 'experimental_orders.mat');
        load(order_file, 'participant_orders');
        current_order = participant_orders(strcmp(participant_orders.ID, id), :).order;
        
        % Define condition order based on experimental design
        if current_order == 1
            conditions_order = [1, 2, 3];  % Full, Partial, No gaze
        elseif current_order == 2
            conditions_order = [2, 1, 3];  % Partial, Full, No gaze
        elseif current_order == 3
            conditions_order = [3, 2, 1];  % No, Partial, Full gaze
        end
        
        % Load audio stimulus onset times
        onset_file = fullfile(base_path, 'stimulus_onsets.mat');
        load(onset_file, 'stim_onsettimes_all');
        
        %% Process each session (typically only 1 session)
        
        for session = 1:1
            % Load EEG data for this participant
            eeg_file = fullfile(eeg_path, [id, '_BABBLE_AR_onset.mat']);
            load(eeg_file, 'FamEEGart');
            
            % Initialize condition matrix for this participant
            CondMatx = cell(session, size(FamEEGart, 1));
            
            %% Process each familiarization block
            
            for block = 1:size(FamEEGart, 1)
                if isempty(FamEEGart{block})
                    continue;
                end
                
                % Initialize storage for this block
                CondMatx{session, block} = struct();
                CondMatx{session, block}.alpha_peak = cell(3, 3);
                CondMatx{session, block}.alpha_lag = cell(3, 3);
                CondMatx{session, block}.theta_peak = cell(3, 3);
                CondMatx{session, block}.theta_lag = cell(3, 3);
                CondMatx{session, block}.delta_peak = cell(3, 3);
                CondMatx{session, block}.delta_lag = cell(3, 3);
                CondMatx{session, block}.xcov_alpha = cell(3, 3);
                CondMatx{session, block}.xcov_theta = cell(3, 3);
                CondMatx{session, block}.xcov_delta = cell(3, 3);
                
                %% Process each condition (gaze type)
                
                for condition = 1:3  % Full, Partial, No gaze
                    %% Process each phrase
                    
                    for phrase = 1:3
                        % Skip if no data available
                        if isempty(FamEEGart{block, 1}) || ...
                           size(FamEEGart{block, 1}, 1) < condition || ...
                           size(FamEEGart{block, 1}, 2) < phrase || ...
                           isempty(FamEEGart{block, 1}{condition, phrase})
                            continue;
                        end
                        
                        %% Load and process EEG data
                        
                        % Get EEG data for current condition/phrase
                        eeg_data = FamEEGart{block, 1}{condition, phrase}';
                        
                        % Skip if insufficient data
                        if size(eeg_data, 2) < 200
                            continue;
                        end
                        
                        % Find syllable markers for first 6 syllables (2 words)
                        syllable_markers = find(eeg_data(33, :) == 999, 7);
                        if length(syllable_markers) < 7
                            continue;
                        end
                        
                        % Extract channels of interest
                        data = eeg_data(roi_channels, :);
                        
                        % Handle artifact and event markers
                        data(data == 999) = NaN;  % Event markers
                        data(data == 888) = NaN;  % Manual rejection
                        data(data == 777) = NaN;  % Unattended periods
                        
                        % Replace NaNs with zeros for filtering
                        data(isnan(data)) = 0;
                        
                        %% Create EEG structure for Hilbert analysis
                        
                        EEGx = struct();
                        EEGx.nbchan = size(data, 1);
                        EEGx.pnts = size(data, 2);
                        EEGx.trials = 1;
                        EEGx.srate = fs;
                        EEGx.event = [];
                        EEGx.data = data;
                        
                        % Extract data for first 6 syllables only
                        syllable_data = data(:, syllable_markers(1):syllable_markers(7) - 1);
                        
                        %% Calculate Hilbert envelopes for each frequency band
                        
                        % Alpha band (6-9 Hz)
                        alpha_hilbert = Hilb_Amp_BABBLE(EEGx, alpha_band);
                        alpha_amplitude = squeeze(abs(alpha_hilbert(1, :, syllable_markers(1):syllable_markers(7) - 1)));
                        alpha_amplitude(syllable_data == 0) = NaN;
                        
                        % Theta band (3-6 Hz)
                        theta_hilbert = Hilb_Amp_BABBLE(EEGx, theta_band);
                        theta_amplitude = squeeze(abs(theta_hilbert(1, :, syllable_markers(1):syllable_markers(7) - 1)));
                        theta_amplitude(syllable_data == 0) = NaN;
                        
                        % Delta band (1-3 Hz)
                        delta_hilbert = Hilb_Amp_BABBLE(EEGx, delta_band);
                        delta_amplitude = squeeze(abs(delta_hilbert(1, :, syllable_markers(1):syllable_markers(7) - 1)));
                        delta_amplitude(syllable_data == 0) = NaN;
                        
                        %% Calculate appropriate lag values for cross-correlation
                        
                        nlags_alpha = round(EEGx.srate / alpha_band(2));
                        nlags_theta = round(EEGx.srate / theta_band(2));
                        nlags_delta = round(EEGx.srate / delta_band(2));
                        
                        %% Load and process corresponding audio stimulus
                        
                        % Get stimulus ID for current condition/phrase
                        audio_file = fullfile(audio_path, sprintf('stimulus_%d_%d.wav', condition, phrase));
                        
                        % Load audio and compute envelope
                        [audio, audio_fs] = audioread(audio_file);
                        audio_envelope = abs(hilbert(audio));
                        
                        % Resample audio envelope to match EEG sampling rate
                        audio_resampled = resample(audio_envelope, fs, audio_fs);
                        
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
                            % Alpha band cross-correlation
                            [xcov_alpha, lags_alpha] = xcov(tiedrank(audio_segment), ...
                                tiedrank(alpha_amplitude(channel, :)'), ...
                                nlags_alpha, 'coeff');
                            
                            % Convert lags to milliseconds
                            lags_alpha_ms = (lags_alpha * 1000) / fs;
                            
                            % Find peak correlation and its lag
                            [peak_alpha, lag_idx_alpha] = max(abs(xcov_alpha));
                            
                            % Store results
                            CondMatx{session, block}.xcov_alpha{condition, phrase}(channel, :) = xcov_alpha;
                            
                            if ~isnan(peak_alpha)
                                CondMatx{session, block}.alpha_peak{condition, phrase}(channel) = peak_alpha;
                                CondMatx{session, block}.alpha_lag{condition, phrase}(channel) = lags_alpha_ms(lag_idx_alpha);
                            else
                                CondMatx{session, block}.alpha_peak{condition, phrase}(channel) = NaN;
                                CondMatx{session, block}.alpha_lag{condition, phrase}(channel) = NaN;
                            end
                            
                            % Theta band cross-correlation
                            [xcov_theta, lags_theta] = xcov(tiedrank(audio_segment), ...
                                tiedrank(theta_amplitude(channel, :)'), ...
                                nlags_theta, 'coeff');
                            
                            % Convert lags to milliseconds
                            lags_theta_ms = (lags_theta * 1000) / fs;
                            
                            % Find peak correlation and its lag
                            [peak_theta, lag_idx_theta] = max(abs(xcov_theta));
                            
                            % Store results
                            CondMatx{session, block}.xcov_theta{condition, phrase}(channel, :) = xcov_theta;
                            
                            if ~isnan(peak_theta)
                                CondMatx{session, block}.theta_peak{condition, phrase}(channel) = peak_theta;
                                CondMatx{session, block}.theta_lag{condition, phrase}(channel) = lags_theta_ms(lag_idx_theta);
                            else
                                CondMatx{session, block}.theta_peak{condition, phrase}(channel) = NaN;
                                CondMatx{session, block}.theta_lag{condition, phrase}(channel) = NaN;
                            end
                            
                            % Delta band cross-correlation
                            [xcov_delta, lags_delta] = xcov(tiedrank(audio_segment), ...
                                tiedrank(delta_amplitude(channel, :)'), ...
                                nlags_delta, 'coeff');
                            
                            % Convert lags to milliseconds
                            lags_delta_ms = (lags_delta * 1000) / fs;
                            
                            % Find peak correlation and its lag
                            [peak_delta, lag_idx_delta] = max(abs(xcov_delta));
                            
                            % Store results
                            CondMatx{session, block}.xcov_delta{condition, phrase}(channel, :) = xcov_delta;
                            
                            if ~isnan(peak_delta)
                                CondMatx{session, block}.delta_peak{condition, phrase}(channel) = peak_delta;
                                CondMatx{session, block}.delta_lag{condition, phrase}(channel) = lags_delta_ms(lag_idx_delta);
                            else
                                CondMatx{session, block}.delta_peak{condition, phrase}(channel) = NaN;
                                CondMatx{session, block}.delta_lag{condition, phrase}(channel) = NaN;
                            end
                        end
                        
                        %% Store additional data for reference
                        
                        CondMatx{session, block}.EEG = EEGx;
                        CondMatx{session, block}.Speech = audio_segment;
                        CondMatx{session, block}.hilb_alpha = alpha_amplitude;
                        CondMatx{session, block}.hilb_theta = theta_amplitude;
                        CondMatx{session, block}.hilb_delta = delta_amplitude;
                        
                        %% Extract peak values and lags for main arrays
                        
                        if ~isempty(CondMatx{session, block}.alpha_peak{condition, phrase})
                            alpha_peak(:, conditions_order(condition), phrase, block, session, pt) = ...
                                CondMatx{session, block}.alpha_peak{condition, phrase};
                            theta_peak(:, conditions_order(condition), phrase, block, session, pt) = ...
                                CondMatx{session, block}.theta_peak{condition, phrase};
                            delta_peak(:, conditions_order(condition), phrase, block, session, pt) = ...
                                CondMatx{session, block}.delta_peak{condition, phrase};
                            
                            % Store lag values (in ms)
                            alpha_lag(:, conditions_order(condition), phrase, block, session, pt) = ...
                                CondMatx{session, block}.alpha_lag{condition, phrase};
                            theta_lag(:, conditions_order(condition), phrase, block, session, pt) = ...
                                CondMatx{session, block}.theta_lag{condition, phrase};
                            delta_lag(:, conditions_order(condition), phrase, block, session, pt) = ...
                                CondMatx{session, block}.delta_lag{condition, phrase};
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
        
        %% Save results for this participant
        
        filename = sprintf('BABBLE_%s_power_xcov_allbands_6syll_filtFULLSET_voltaverage_W9chans.mat', id);
        save(fullfile(results_path, filename), 'CondMatx', '-v7.3');
        
        % Clear large variables to save memory
        clear CondMatx syllable_markers
    end
end

% Note: The Hilb_Amp_BABBLE function is assumed to be available in the path
% This function computes the Hilbert transform in the specified frequency band

%% =================== PART 2: PROCESS AND ANALYZE ENTRAINMENT DATA ===================

clear all;
clc;

% Set base path
base_path = '/path/to/data/';
results_path = fullfile(base_path, 'entrainment_results');

%% Process data for both locations and combine

for location = 1:2
    % Set location-specific parameters
    if location == 1
        location_name = 'UK';
        participant_files = dir(fullfile(results_path, 'BABBLE_P1*_power_xcov*.mat'));
    else
        location_name = 'SG';  
        participant_files = dir(fullfile(results_path, 'BABBLE_P2*_power_xcov*.mat'));
    end
    
    % Extract participant IDs
    participant_ids = {participant_files.name}';
    participant_ids = cellfun(@(x) extractBetween(x, 'BABBLE_', '_power'), participant_ids, 'UniformOutput', false);
    participant_ids = [participant_ids{:}]';
    
    fprintf('Processing %d participants from %s\n', length(participant_ids), location_name);
    
    % Define channel names
    channels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
    
    % Initialize storage for all entrainment data
    data = [];
    ids = {};
    
    %% Process each participant
    
    for pt = 1:length(participant_ids)
        % Load entrainment data
        filename = sprintf('BABBLE_%s_power_xcov_allbands_6syll_filtFULLSET_voltaverage_W9chans.mat', participant_ids{pt});
        load(fullfile(results_path, filename), 'CondMatx');
        
        % Process each session, block, condition, and phrase
        for session = 1:1
            for block = 1:size(CondMatx, 2)
                for condition = 1:3
                    for phrase = 1:3
                        % Skip empty data
                        if isempty(CondMatx{session, block}) || ...
                           isempty(CondMatx{session, block}.alpha_peak) || ...
                           isempty(CondMatx{session, block}.alpha_peak{condition, phrase})
                            continue;
                        end
                        
                        % Extract entrainment values for all channels
                        alpha_peak_vals = CondMatx{session, block}.alpha_peak{condition, phrase};
                        alpha_lag_vals = CondMatx{session, block}.alpha_lag{condition, phrase};
                        theta_peak_vals = CondMatx{session, block}.theta_peak{condition, phrase};
                        theta_lag_vals = CondMatx{session, block}.theta_lag{condition, phrase};
                        delta_peak_vals = CondMatx{session, block}.delta_peak{condition, phrase};
                        delta_lag_vals = CondMatx{session, block}.delta_lag{condition, phrase};
                        
                        % Create row with all metrics
                        new_row = [
                            block, condition, phrase, ...
                            alpha_peak_vals', alpha_lag_vals', ...
                            theta_peak_vals', theta_lag_vals', ...
                            delta_peak_vals', delta_lag_vals'
                        ];
                        
                        % Handle missing data (convert zeros to NaN if too many zeros)
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
    if location == 1
        result_table_uk = result_table;
    else
        result_table_sg = result_table;
    end
end

%% Combine tables from both locations

table_combined = [result_table_uk; result_table_sg];

% Save combined table
save(fullfile(base_path, 'entrainment_table.mat'), 'table_combined');
writetable(table_combined, fullfile(base_path, 'entrainment_table.xlsx'));

fprintf('Neural entrainment analysis complete. Results saved to entrainment_table.mat and .xlsx\n');
