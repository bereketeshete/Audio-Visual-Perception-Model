
% Preprocess data

function [logmel, train_video] = preprocess_data(train_audio_dir, train_video_dir, train_prep_audio_dir, train_prep_video_dir)

train_audio = load_data(train_audio_dir);
train_video = load_data(train_video_dir);
num_data = size(train_audio,1);     % number of the training data

%% STEP 2. Audio pre-processing

% pre-processing parameters [DO NOT MODIFY]
opt.fs = 48000;             % frequency sampling rates
opt.frameLenMS = 25;        % Frame length in ms
opt.frameShiftMS = 10;      % Frame shift in ms
opt.preemph = 0.97;         % Pre-emphasize parameter
opt.minfreq = 0;           % Minimum frequency
opt.maxfreq = 8000;         % Maximum frequency
opt.nfilts = 26;            % Number of Mel filters
opt.draw_figure = false;

% extracting log-mel spectrogram
logmel = cell(num_data,1);
for idx = 1:num_data
    logmel{idx,1} = filterbank(train_audio{idx,1}, opt);                    %% Task 1. fill in FILL_filterbank.m
end

save(train_prep_audio_dir, 'logmel');


%% STEP 3. Image pre-processing

patch_width=48;   % width of the patches [DO NOT MODIFY]
patch_height=96;  % height of the patches [DO NOT MODIFY]

train_video = normalize_image(train_video, patch_width, patch_height);      %% Task 2. fill in FILL_normalize_image.m
save(train_prep_video_dir, 'train_video');


end
