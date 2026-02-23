%% MFCC Feature Extractor for Audio Datasets
% This script extracts 33-dimensional MFCCs features from .wav files,
% applies pre-emphasis, and saves the results in JSON format.
%
% Requirements: Audio Toolbox, AMT (Auditory Modeling Toolbox)
% Author: [Ikumi Saito]
% Date: 2026.2.20 (Updated for Public Release)

clear;
amt_start('silent'); % Initialize Auditory Modeling Toolbox
tic;

%% --- Configuration ---
% Define your input and output directories
input_dir  = 'path/to/your/input_audio';  % Path to source .wav files
output_dir = 'path/to/output_features';  % Path to save extracted JSONs

% Signal processing parameters
PRE_EMPHASIS_COEFF = 0.97;
WIN_DURATION_MS    = 0.030; % 30ms window
OVERLAP_RATIO      = 0.5;   % 50% overlap (15ms)
NUM_MEL_BANDS      = 40;
NUM_MFCC_COEFFS    = 33;    % Matched with GFCC dimensions for consistency
TRUNCATE_PRECISION = 6;     % Decimal places for output stability

% Variables to persist through the cleaning process in the loop
PERM_VARS = {
    'input_dir', 'output_dir', 'fileinfo', 'numFiles', 'PERM_VARS', ...
    'PRE_EMPHASIS_COEFF', 'WIN_DURATION_MS', 'OVERLAP_RATIO', ...
    'NUM_MEL_BANDS', 'NUM_MFCC_COEFFS', 'TRUNCATE_PRECISION', ...
    'skipped_log', 'warning_log'
};

%% --- Initialization ---
fileinfo = dir(fullfile(input_dir, '*.wav'));
numFiles = length(fileinfo);

if numFiles == 0
    error('No .wav files found in the target directory: %s', input_dir);
end

% Initialize logs for errors and warnings
skipped_log = struct('files', {{}}, 'messages', {{}}, 'count', 0);
warning_log = struct('files', {{}}, 'messages', {{}}, 'count', 0);

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
        %% 1. Load Audio File
        filepath_in = fullfile(fileinfo(fileid).folder, current_filename);
        [input_raw, fs] = audioread(filepath_in);
        
        % Check if the file is long enough for the processing window
        win_size_samples = floor(fs * WIN_DURATION_MS);
        if size(input_raw, 1) < win_size_samples
            error('File length is shorter than the window size (%d samples).', win_size_samples);
        end

        %% 2. Pre-processing
        % Apply pre-emphasis filter to boost high frequencies
        input_preemphasized = filter([1, -PRE_EMPHASIS_COEFF], 1, input_raw);

        %% 3. MFCCs Extraction
        lastwarn(''); % Reset warning state

        % Compute Mel-spectrogram (Range: 25Hz to 8000Hz)
        S = melSpectrogram(input_preemphasized, fs, ...
            'Window', hamming(win_size_samples), ...
            'OverlapLength', floor(win_size_samples * OVERLAP_RATIO), ...
            'NumBands', NUM_MEL_BANDS, ...
            'FrequencyRange', [25 8000]);

        % Apply Log and Discrete Cosine Transform (DCT)
        % Note: Using log10 for specific research consistency
        coeffs = dct(log10(S + eps));
        
        % Extract first N coefficients and transpose for time-major format
        coeffs = coeffs(1:NUM_MFCC_COEFFS, :)'; 

        % Check for numerical warnings during computation
        [warnMsg, ~] = lastwarn;
        if ~isempty(warnMsg)
            warning_log.count = warning_log.count + 1;
            warning_log.files{warning_log.count} = current_filename;
            warning_log.messages{warning_log.count} = strtrim(warnMsg);
        end

        %% 4. Data Post-processing
        % Truncate decimals to control precision and file size
        precision_factor = 10^-TRUNCATE_PRECISION;
        coeffs = coeffs - rem(coeffs, precision_factor);

        % Prepare structure for JSON serialization
        for i = 1:NUM_MFCC_COEFFS
            s.(sprintf("mfcc_%02d", i)) = coeffs(:, i);
        end

        processed_ok = true;

    catch ME
        fprintf("  --> ERROR: %s\n", ME.message);
        skipped_log.count = skipped_log.count + 1;
        skipped_log.files{skipped_log.count} = current_filename;
        skipped_log.messages{skipped_log.count} = ME.message;
    end

    %% 5. Export Data
    if processed_ok
        filename_base = current_filename(1:end-4); % Remove .wav extension
        filepath_out = fullfile(output_dir, [filename_base, '.json']);
        writestruct(s, filepath_out, "PrettyPrint", false);
    end

    % Memory management: Clear temporary variables except configurations
    clearvars('-except', PERM_VARS{:});
end

%% --- Final Summary ---
fprintf("\n" + repmat('-', 1, 50) + "\n");
fprintf("Extraction Complete!\n");
fprintf("Total processed: %d\n", numFiles - skipped_log.count);
fprintf("Total skipped  : %d\n", skipped_log.count);

if skipped_log.count > 0
    for i = 1:skipped_log.count
        fprintf("  [Skip] %s: %s\n", skipped_log.files{i}, skipped_log.messages{i});
    end
end

if warning_log.count > 0
    fprintf("Total warnings : %d\n", warning_log.count);
end
fprintf(repmat('-', 1, 50) + "\n");
toc;