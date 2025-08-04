%% Surrogate Neural Entrainment Analysis
% All datasets have been made publicly available through Nanyang Technological University (NTU)'s 
% data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to 
% NTU's open access policy.
%
% Purpose: Generate surrogate data for neural-speech entrainment significance testing
% by randomly disrupting temporal dependencies between signals
%
% This script:
% 1. Creates permutation-based surrogate datasets for entrainment analysis
% 2. Randomly shuffles temporal alignment between EEG and speech envelopes
% 3. Preserves spectral content while disrupting temporal relationships
% 4. Generates null distribution for statistical significance testing

%% =================== PART 1: GENERATE SURROGATE ENTRAINMENT DATA ===================

clc;
clear all;

% Set base paths
base_path = '/path/to/data/';
results_path = fullfile(base_path, 'entrainment_results');
surrogate_path = fullfile(base_path, 'surrogate_entrainment');
audio_path = fullfile(base_path, 'audio_files');

% Number of surrogate iterations
num_surrogates = 1000;

fprintf('Starting surrogate entrainment analysis with %d iterations\n', num_surrogates);

%% Generate surrogate data for each iteration

for surr_idx = 1:num_surrogates
    fprintf('Processing surrogate iteration %d of %d\n', surr_idx, num_surrogates);
    
    % Create directory for this surrogate iteration
    surr_output_dir = fullfile(surrogate_path, sprintf('permu%d', surr_idx));
    if ~exist(surr_output_dir, 'dir')
        mkdir(surr_output_dir);
    end
    
    %% Process data for both locations
    
    for location = 1:2
        % Set location-specific parameters
        if location == 1
            location_name = 'UK';
            sg = 0;
            data_dir = fullfile(base_path, 'Preprocessed_Data_camb/');
            participant_files = dir(fullfile(data_dir, 'P*/*_BABBLE_AR.mat'));
        else
            location_name = 'SG';
            sg = 1;
            data_dir = fullfile(base_path, 'Preprocessed_Data_sg/');
            participant_files = dir(fullfile(data_dir, 'P*/P1*_BABBLE_AR.mat'));
        end
        
        % Extract participant IDs
        participant_ids = {participant_files.name}';
        
        %% Define analysis parameters
        
        % EEG sampling rate
        fs = 200;  % Hz
        
        % Frequency bands for entrainment analysis
        delta_band = [1, 3];   % Delta: 1-3 Hz
        theta_band = [3, 6];   % Theta: 3-6 Hz
        alpha_band = [6, 9];   % Alpha: 6-9 Hz
        
        % EEG channels of interest
        roi_channels = [4:6, 15:17, 26:28];  % F3,Fz,F4,C3,Cz,C4,P3,Pz,P4
        channels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
        
        %% Process each participant
        
        for pt = 1:length(participant_ids)
            id = participant_ids{pt};
            if sg == 1
                participant_name = ['P0', id(3:5)];
            else
                participant_name = id(1:5);
            end
            
            fprintf('  Processing participant %s\n', participant_name);
            
            % Load experimental order information
            order_file = fullfile(base_path, 'experimental_orders.mat');
            load(order_file, 'participant_orders');
            current_order = participant_orders(...
                strcmp(participant_orders.ID, participant_name), :).order;
            
            % Define condition order based on experimental design
            if current_order == 1
                conditions_order = [1, 2, 3];  % Full, Partial, No gaze
            elseif current_order == 2
                conditions_order = [2, 1, 3];  % Partial, Full, No gaze
            elseif current_order == 3
                conditions_order = [3, 2, 1];  % No, Partial, Full gaze
            end
            
            % Load stimulus information
            babble_file = fullfile(base_path, 'babble_orders.txt');
            fid = fopen(babble_file);
            babble_orders = textscan(fid, '%f%f', 'delimiter', '\t');
            fclose(fid);
            
            babble_ppts = babble_orders{1,1};
            babble_ppts_orders = babble_orders{1,2};
            
            % Get randomized order for surrogate analysis
            randomized_order = sprintf('order%d', babble_ppts_orders(...
                babble_ppts == str2num(participant_name(3:5))));
            
            % Load stimulus IDs and onset times
            load(fullfile(base_path, 'stimulus_info.mat'), 'stim_ids_all');
            load(fullfile(base_path, 'downsampled_onset.mat'), 'ds_onset');
            stim_onsettimes_all = ds_onset{babble_ppts_orders(...
                babble_ppts == str2num(participant_name(3:5))), 1};
            
            % Define gaze condition mapping based on randomized order
            if strcmp(randomized_order, 'order1')
                gaze_order_for_wav = [1, 2, 3];  % Full, Partial, No gaze
            elseif strcmp(randomized_order, 'order2')
                gaze_order_for_wav = [3, 1, 2];  % No, Full, Partial gaze
            elseif strcmp(randomized_order, 'order3')
                gaze_order_for_wav = [2, 3, 1];  % Partial, No, Full gaze
            end
            
            %% Process each session
            
            for session = 1:1  % Typically only one session
                % Load EEG data for this participant
                eeg_file = fullfile(data_dir, participant_name, ...
                    [participant_name, '_BABBLE_AR_onset.mat']);
                load(eeg_file, 'FamEEGart');
                
                % Initialize condition matrix for surrogate data
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
                    
                    %% Process each condition
                    
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
                            
                            %% SURROGATE METHODOLOGY: Randomly select different audio
                            % For surrogate analysis, randomly pair EEG with different audio
                            random_condition = randi(3);
                            random_phrase = randi(3);
                            
                            %% Load and process EEG data (real)
                            
                            eeg_data = FamEEGart{block, 1}{condition, phrase}';
                            
                            % Skip if insufficient data
                            if size(eeg_data, 2) < 200
                                continue;
                            end
                            
                            % Find syllable markers
                            syllable_markers = find(eeg_data(33, :) == 999, 7);
                            if length(syllable_markers) < 7
                                continue;
                            end
                            
                            % Extract channels of interest
                            data = eeg_data(roi_channels, :);
                            
                            % Handle artifact markers
                            data(data == 999) = NaN;  % Event markers
                            data(data == 888) = NaN;  % Manual rejection
                            data(data == 777) = NaN;  % Unattended periods
                            
                            % Replace NaNs with zeros for filtering
                            data(isnan(data)) = 0;
                            
                            %% Create EEG structure
                            
                            EEGx = struct();
                            EEGx.nbchan = size(data, 1);
                            EEGx.pnts = size(data, 2);
                            EEGx.trials = 1;
                            EEGx.srate = fs;
                            EEGx.event = [];
                            EEGx.data = data;
                            
                            % Extract data for first 6 syllables
                            syllable_data = data(:, syllable_markers(1):syllable_markers(7) - 1);
                            
                            %% Calculate Hilbert envelopes for EEG
                            
                            % Alpha band (6-9 Hz)
                            alpha_hilbert = Hilb_Amp_BABBLE(EEGx, alpha_band);
                            alpha_amplitude = squeeze(abs(alpha_hilbert(1, :, ...
                                syllable_markers(1):syllable_markers(7) - 1)));
                            alpha_amplitude(syllable_data == 0) = NaN;
                            
                            % Theta band (3-6 Hz)
                            theta_hilbert = Hilb_Amp_BABBLE(EEGx, theta_band);
                            theta_amplitude = squeeze(abs(theta_hilbert(1, :, ...
                                syllable_markers(1):syllable_markers(7) - 1)));
                            theta_amplitude(syllable_data == 0) = NaN;
                            
                            % Delta band (1-3 Hz)
                            delta_hilbert = Hilb_Amp_BABBLE(EEGx, delta_band);
                            delta_amplitude = squeeze(abs(delta_hilbert(1, :, ...
                                syllable_markers(1):syllable_markers(7) - 1)));
                            delta_amplitude(syllable_data == 0) = NaN;
                            
                            %% Calculate lag values for cross-correlation
                            
                            nlags_alpha = round(EEGx.srate / alpha_band(2));
                            nlags_theta = round(EEGx.srate / theta_band(2));
                            nlags_delta = round(EEGx.srate / delta_band(2));
                            
                            %% Load RANDOMIZED audio for surrogate analysis
                            
                            % Use randomly selected condition and phrase for audio
                            audio_file = fullfile(audio_path, sprintf('stimulus_%d_%d.wav', ...
                                random_condition, random_phrase));
                            
                            % Load audio and compute envelope
                            [audio, audio_fs] = audioread(audio_file);
                            audio_envelope = abs(hilbert(audio));
                            
                            % Resample to EEG sampling rate
                            audio_resampled = resample(audio_envelope, fs, audio_fs);
                            
                            %% Extract speech envelope segment
                            
                            current_wav_onset = stim_onsettimes_all{random_condition, random_phrase};
                            onset_idx = find(current_wav_onset == 999, 1);
                            
                            if isempty(onset_idx)
                                continue;
                            end
                            
                            segment_length = syllable_markers(7) - syllable_markers(1);
                            audio_segment = audio_resampled(onset_idx:(onset_idx + segment_length - 1));
                            
                            %% ADDITIONAL SURROGATE STEP: Phase randomization
                            % Apply phase randomization to further disrupt temporal relationships
                            audio_fft = fft(audio_segment);
                            phase_randomized = abs(audio_fft) .* exp(1i * 2 * pi * rand(size(audio_fft)));
                            audio_segment = real(ifft(phase_randomized));
                            
                            %% Calculate cross-correlation with randomized audio
                            
                            for channel = 1:EEGx.nbchan
                                % Alpha band
                                [xcov_alpha, lags_alpha] = xcov(tiedrank(audio_segment), ...
                                    tiedrank(alpha_amplitude(channel, :)'), ...
                                    nlags_alpha, 'coeff');
                                
                                lags_alpha_ms = (lags_alpha * 1000) / fs;
                                [peak_alpha, lag_idx_alpha] = max(abs(xcov_alpha));
                                
                                CondMatx{session, block}.xcov_alpha{condition, phrase}(channel, :) = xcov_alpha;
                                
                                if ~isnan(peak_alpha)
                                    CondMatx{session, block}.alpha_peak{condition, phrase}(channel) = peak_alpha;
                                    CondMatx{session, block}.alpha_lag{condition, phrase}(channel) = ...
                                        lags_alpha_ms(lag_idx_alpha);
                                else
                                    CondMatx{session, block}.alpha_peak{condition, phrase}(channel) = NaN;
                                    CondMatx{session, block}.alpha_lag{condition, phrase}(channel) = NaN;
                                end
                                
                                % Theta band
                                [xcov_theta, lags_theta] = xcov(tiedrank(audio_segment), ...
                                    tiedrank(theta_amplitude(channel, :)'), ...
                                    nlags_theta, 'coeff');
                                
                                lags_theta_ms = (lags_theta * 1000) / fs;
                                [peak_theta, lag_idx_theta] = max(abs(xcov_theta));
                                
                                CondMatx{session, block}.xcov_theta{condition, phrase}(channel, :) = xcov_theta;
                                
                                if ~isnan(peak_theta)
                                    CondMatx{session, block}.theta_peak{condition, phrase}(channel) = peak_theta;
                                    CondMatx{session, block}.theta_lag{condition, phrase}(channel) = ...
                                        lags_theta_ms(lag_idx_theta);
                                else
                                    CondMatx{session, block}.theta_peak{condition, phrase}(channel) = NaN;
                                    CondMatx{session, block}.theta_lag{condition, phrase}(channel) = NaN;
                                end
                                
                                % Delta band
                                [xcov_delta, lags_delta] = xcov(tiedrank(audio_segment), ...
                                    tiedrank(delta_amplitude(channel, :)'), ...
                                    nlags_delta, 'coeff');
                                
                                lags_delta_ms = (lags_delta * 1000) / fs;
                                [peak_delta, lag_idx_delta] = max(abs(xcov_delta));
                                
                                CondMatx{session, block}.xcov_delta{condition, phrase}(channel, :) = xcov_delta;
                                
                                if ~isnan(peak_delta)
                                    CondMatx{session, block}.delta_peak{condition, phrase}(channel) = peak_delta;
                                    CondMatx{session, block}.delta_lag{condition, phrase}(channel) = ...
                                        lags_delta_ms(lag_idx_delta);
                                else
                                    CondMatx{session, block}.delta_peak{condition, phrase}(channel) = NaN;
                                    CondMatx{session, block}.delta_lag{condition, phrase}(channel) = NaN;
                                end
                            end
                            
                            %% Store reference data
                            
                            CondMatx{session, block}.EEG = EEGx;
                            CondMatx{session, block}.Speech = audio_segment;
                            CondMatx{session, block}.hilb_alpha = alpha_amplitude;
                            CondMatx{session, block}.hilb_theta = theta_amplitude;
                            CondMatx{session, block}.hilb_delta = delta_amplitude;
                        end
                    end
                end
            end
            
            %% Save surrogate results for this participant
            
            filename = sprintf('%s_permu%d.mat', participant_name, surr_idx);
            save(fullfile(surr_output_dir, filename), 'CondMatx', '-v7.3');
            
            % Clear large variables
            clear CondMatx syllable_markers
        end
    end
    
    fprintf('Completed surrogate iteration %d of %d\n', surr_idx, num_surrogates);
end

%% =================== PART 2: COMPILE SURROGATE DATA ===================

clear all;
clc;

% Set base path
base_path = '/path/to/data/';
surrogate_path = fullfile(base_path, 'surrogate_entrainment');
permutable = cell(1000, 1);  % Storage for surrogate datasets

% Number of surrogate iterations to process
num_surrogates = 1000;

fprintf('Compiling surrogate data from %d iterations\n', num_surrogates);

%% Process each surrogate iteration

for surr_idx = 1:num_surrogates
    fprintf('Processing surrogate %d of %d\n', surr_idx, num_surrogates);
    
    % Initialize combined data for this iteration
    combined_data = [];
    combined_ids = {};
    
    %% Process both locations
    
    for location_idx = 1:2
        % Set location-specific parameters
        if location_idx == 2
            sg = 1;  % Singapore
            data_dir = fullfile(base_path, 'Preprocessed_Data_sg/');
            list_ppts = dir(fullfile(data_dir, 'P*/P1*_BABBLE_AR.mat'));
        else
            sg = 0;  % Cambridge
            data_dir = fullfile(base_path, 'Preprocessed_Data_camb/');
            list_ppts = dir(fullfile(data_dir, 'P*/*_BABBLE_AR.mat'));
        end
        
        % Define surrogate data path
        surr_path = fullfile(surrogate_path, sprintf('permu%d', surr_idx));
        
        participant_ids = {list_ppts.name}';
        
        % Define channel names
        channels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'};
        
        % Initialize data arrays for this location
        data = [];
        ids = {};
        
        %% Process each participant
        
        for pt = 1:length(participant_ids)
            % Get participant name
            p_name = participant_ids{pt};
            if sg == 1
                p_name = ['P0', p_name(3:5)];
            else
                p_name = p_name(1:5);
            end
            
            % Load surrogate data for this participant
            filename = fullfile(surr_path, sprintf('%s_permu%d.mat', p_name, surr_idx));
            if ~exist(filename, 'file')
                continue;
            end
            
            load(filename, 'CondMatx');
            
            % Extract entrainment measures
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
                            
                            % Extract entrainment values
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
                            
                            % Handle missing data
                            if length(find(new_row == 0)) < 5
                                data = [data; new_row];
                                ids = [ids; {p_name}];
                            else
                                new_row(new_row == 0) = NaN;
                                data = [data; new_row];
                                ids = [ids; {p_name}];
                            end
                        end
                    end
                end
            end
        end
        
        % Add location data to combined dataset
        combined_data = [combined_data; data];
        combined_ids = [combined_ids; ids];
    end
    
    %% Create table for this surrogate iteration
    
    if ~isempty(combined_data)
        % Define column names
        column_names = {'block', 'condition', 'phrase'};
        metrics = {'alpha_peak', 'alpha_lag', 'theta_peak', 'theta_lag', 'delta_peak', 'delta_lag'};
        
        for i = 1:length(metrics)
            for j = 1:length(channels)
                column_names = [column_names, sprintf('%s_%s', metrics{i}, channels{j})];
            end
        end
        
        % Create table
        result_table = array2table(combined_data, 'VariableNames', column_names);
        
        % Add participant IDs
        result_table = addvars(result_table, combined_ids, 'Before', 'block', ...
            'NewVariableNames', {'ID'});
        
        % Store in permutable cell array
        permutable{surr_idx} = result_table;
    end
end

%% Save compiled surrogate data

save(fullfile(base_path, 'ENTRIANSURR.mat'), 'permutable', '-v7.3');

fprintf('Surrogate entrainment analysis complete. Results saved to ENTRIANSURR.mat\n');
