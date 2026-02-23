%% GFCC Feature Extractor for Audio Datasets
% This script extracts Gammatone Frequency Cepstral Coefficients (GFCCs) 
% from .wav files, incorporating pre-emphasis and log-energy replacement.
%
% Requirements: Audio Toolbox, AMT (Auditory Modeling Toolbox)
% Author: [Ikumi Saito]
% Date: 2025.2.20 (Updated for Public Release)

clear;
amt_start('silent'); % Initialize Auditory Modeling Toolbox
tic;

%% --- Configuration ---
% Define your input and output directories
%input_dir  = 'path/to/your/input_audio';  % Path to source .wav files
%output_dir = 'path/to/output_features';  % Path to save extracted JSONs
input_dir  = '/Users/samusushi/Mlearn/DatasetML/Dataset_22k/ALL_Dataset_22k';  % Path to source .wav files
output_dir = '/Users/samusushi/Mlearn/DatasetML/Extracted_Features/Comp_GFCC_ALL_Dataset_22k'; 

% Signal processing parameters
PRE_EMPHASIS_COEFF = 0.97;
WIN_DURATION_MS    = 0.030; % 30ms window
HOP_DURATION_MS    = 0.015; % 15ms hop (50% overlap)
TRUNCATE_PRECISION = 6;     % Decimal places for output stability

% Gammatone Filterbank parameters
FLOW  = 25;   % Lower frequency bound (Hz)
FHIGH = 8000; % Upper frequency bound (Hz). 
              % Note: Can be increased up to Nyquist (fs/2) if high-freq content is critical.

% Variables to persist through the cleaning process in the loop
PERM_VARS = {
    'input_dir', 'output_dir', 'fileinfo', 'numFiles', 'PERM_VARS', ...
    'PRE_EMPHASIS_COEFF', 'WIN_DURATION_MS', 'HOP_DURATION_MS', ...
    'TRUNCATE_PRECISION', 'FLOW', 'FHIGH', 'skipped_log'
};

%% --- Initialization ---
fileinfo = dir(fullfile(input_dir, '*.wav'));
numFiles = length(fileinfo);

if numFiles == 0
    error('No .wav files found in the target directory: %s', input_dir);
end

% Initialize log for skipped files
skipped_log = struct('files', {{}}, 'count', 0);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% --- Main Processing Loop ---
for fileid = 1:numFiles
    current_filename = fileinfo(fileid).name;
    fprintf("Processing [%d/%d]: %s\n", fileid, numFiles, current_filename);
    
    s = struct(); % Output structure for JSON
    processed_ok = false;

    try
        %% 1. Load Audio and Pre-process
        filepath_in = fullfile(fileinfo(fileid).folder, current_filename);
        [input_raw, fs] = audioread(filepath_in);
        
        % Apply pre-emphasis filter
        input_filtered = filter([1, -PRE_EMPHASIS_COEFF], 1, input_raw);
        file_size = size(input_filtered, 1);

        %% 2. Framing Parameters
        win_size_samples = floor(fs * WIN_DURATION_MS);
        hop_size_samples = floor(fs * HOP_DURATION_MS);
        win = hamming(win_size_samples);

        % Skip file if shorter than the required window size
        if file_size < win_size_samples
            error('File length (%d) is shorter than window size (%d).', file_size, win_size_samples);
        end
        
        % Calculate number of frames
        num_frames = floor((file_size - win_size_samples) / hop_size_samples) + 1;

        %% 3. Gammatone Filterbank Analysis
        % Generate filterbank coefficients using AMT
        [b, a] = gammatone(erbspacebw(FLOW, FHIGH), fs, 'complex');
        
        % Apply filterbank to the entire signal
        gammaout = 2 * real(ufilterbankz(b, a, input_filtered));
        [~, num_channels] = size(gammaout);

        %% 4. Frame-wise Energy Calculation
        log_energy = zeros(num_frames, num_channels);
        frame_log_energy_total = zeros(num_frames, 1);
        
        for i = 1:num_frames
            sidx = (i-1) * hop_size_samples + 1;
            eidx = sidx + win_size_samples - 1;
            
            if eidx > file_size, break; end
        
            % Calculate C0 (Log-energy of the original filtered frame)
            raw_frame_win = input_filtered(sidx:eidx) .* win;
            frame_log_energy_total(i) = log(sum(raw_frame_win.^2) + eps);
        
            % Calculate energy for each Gammatone channel
            gamma_frame_win = gammaout(sidx:eidx, :) .* repmat(win, 1, num_channels);
            frame_energy = sum(gamma_frame_win.^2, 1); 
            log_energy(i, :) = log(frame_energy + eps);
        end

        %% 5. GFCC Computation (DCT)
        % Apply Discrete Cosine Transform to log-energies
        gfcc = dct(log_energy')'; 
        
        % Replace the first coefficient (C1) with the total log energy (C0 logic)
        gfcc(:, 1) = frame_log_energy_total;
        
        % Precision truncation for numerical consistency
        precision_factor = 10^-TRUNCATE_PRECISION;
        gfcc = gfcc - rem(gfcc, precision_factor);

        %% 6. Prepare Data for Export
        [~, num_coeffs] = size(gfcc); 
        for i = 1:num_coeffs
            s.(sprintf('gfcc_%02d', i)) = gfcc(:, i);
        end

        processed_ok = true;

    catch ME
        fprintf("  --> SKIPPING: %s\n", ME.message);
        skipped_log.count = skipped_log.count + 1;
        skipped_log.files{skipped_log.count} = current_filename;
    end

    %% 7. Export Data
    if processed_ok
        filename_base = current_filename(1:end-4);
        filepath_out = fullfile(output_dir, [filename_base, '.json']);
        writestruct(s, filepath_out, "PrettyPrint", false);
    end

    % Memory management
    clearvars('-except', PERM_VARS{:});
end

%% --- Final Summary ---
fprintf("\n" + repmat('-', 1, 50) + "\n");
fprintf("GFCC Extraction Complete!\n");
if skipped_log.count > 0
    fprintf("Total files skipped: %d\n", skipped_log.count);
    for i = 1:skipped_log.count
        fprintf("  - %s\n", skipped_log.files{i});
    end
else
    fprintf("No files were skipped.\n");
end
fprintf(repmat('-', 1, 50) + "\n");
toc;