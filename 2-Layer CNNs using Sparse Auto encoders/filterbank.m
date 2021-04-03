
function logmel = filterbank(samples, opt)
%%
samples = samples(:,1);
samples=samples/max(abs(samples));

% play the sound
% gong = audioplayer(samples, opt.opt.fs);
% play(gong);

% ---------  Parameters  --------------------
if (opt.fs ~= 16000)
    samples = resample(samples, 16000, opt.fs);
    fs = 16000;
end

frameLen = round(opt.frameLenMS * fs/1000);
frameShift = round(opt.frameShiftMS * fs/1000);

% ---------  Pre-emphasize  ---------------------
samples = filter([1 -opt.preemph], 1, samples);

% calculate spectrogram
nfft = 2^(ceil(log(frameLen)/log(2)));
WINDOW = hamming(frameLen);
NOVERLAP = frameLen - frameShift;

[S,F,T,y] = spectrogram(samples, WINDOW, NOVERLAP, nfft, fs);

if (opt.draw_figure)    % plot spectrogram
    figure(1);
    imagesc(T',F,log(y)); set(gca, 'yDir', 'normal');
    title('Spectrogram');  xlabel('Time (sec)'); ylabel('Frequency (Hz)'); colorbar;
    drawnow;
end

% ---------  Mel Filters  -----------------------
nfreqs = size(y, 1);
nfilts = opt.nfilts;
MelFilters = zeros(nfilts, nfft);
fftfrqs = [0:nfft-1] / nfft * fs; % Center freqs of each FFT bin
minfreqMel = 2595 * log10(1 + (opt.minfreq/700)); %Hz to Mel
maxfreqMel = 2595 * log10(1 + (opt.maxfreq/700)); %Hz to Mel

binfrqsMel = minfreqMel + [0:(nfilts+1)]/(nfilts+1)*(maxfreqMel-minfreqMel);% centers are linear in Mel domain
binfrqs = 700*(10.^(binfrqsMel/2595)-1);% Mel to Hz

for i = 1:nfilts
    fr = binfrqs(i+[0 1 2]);
    loslope = (fftfrqs - fr(1))/(fr(2) - fr(1));% upward lope
    hislope = (fr(3) - fftfrqs)/(fr(3) - fr(2));%downward lope
    MelFilters(i,:) = max(0,min(loslope, hislope));
end
MelFilters = MelFilters(:, 1:nfreqs);

if (opt.draw_figure)    % plot Mel filters
    figure(2);
    for i=1:nfilts
        plot(F, MelFilters(i, :)); hold on;
    end
    set(gca, 'yDir', 'normal');
    title('Mel Filters');xlabel('Frequency (Hz)'); ylabel('Arbitrary Scale');
    drawnow;
end

logmel = MelFilters * y; % filtering
logmel=log10(logmel+10^-30);

if (opt.draw_figure)    % plot logMel features
    figure(3); MF=1:nfilts;
    imagesc(T',MF,logmel);set(gca, 'yDir', 'normal'); title('Mel-Spectrogram');  xlabel('Time (sec)'); ylabel('MelFilter Index'); colorbar;
    drawnow;
end

