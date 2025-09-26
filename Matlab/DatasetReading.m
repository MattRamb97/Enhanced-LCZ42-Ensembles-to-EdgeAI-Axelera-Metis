%download the dataset from
%https://dataserv.ub.tum.de/index.php/s/m1483140

% DATASETREADING  So2Sat LCZ42 loader with robust preprocessing and splits.
%
% [dsTrain, dsCal, dsTest, info] = DatasetReading(cfg)
%
% INPUT (struct cfg) — REQUIRED FIELDS MARKED *
%   *cfg.trainTable   : table with variables  {'Path','Label','CityID','Modality'}
%   *cfg.testTable    : table with variables  {'Path','Label','CityID','Modality'}
%                       Path   -> path to a .mat/.npy/.tif patch or a folder
%                       Label  -> categorical or uint8 in [1..C]
%                       CityID -> string/char/int, lets us do city-wise splits
%                       Modality -> 'MS' (Sentinel-2) or 'SAR' (Sentinel-1)
%
%   %% Preprocessing (all optional, sensible defaults)
%   cfg.inputSize        = [32 32];      % patch spatial size (H W)
%   cfg.useZscore        = false;        % per-band z-score on top of paper scaling
%   cfg.useSARdespeckle  = false;        % non-local means on SAR (mild)
%   cfg.useAugmentation  = true;         % aligned geometric augs (train only)
%   cfg.calibrationFrac  = 0.08;         % fraction of TRAIN cities for calibration
%   cfg.randomSeed       = 42;
%
%   %% I/O
%   cfg.reader.type      = 'auto';       % 'auto'|'mat'|'tif'|'custom'
%   cfg.reader.field     = 'patch';      % for MAT: variable name
%   cfg.reader.customFcn = [];           % @(path) -> HxWxB single
%
% OUTPUT
%   dsTrain : datastore yielding {X,label,cityID,modality} with preprocessing/augs
%   dsCal   : calibration datastore (NO augmentation)
%   dsTest  : test datastore (NO augmentation)
%   info    : struct with statistics and helpers:
%             .bands.mu/.sigma (per-band), .classWeights, .numClasses,
%             .trainCities, .calCities, .testCities, .preprocessFcn, etc.
%
% NOTES
% - This loader preserves the professor's normalization:
%   * Multispectral (MS): scale to [0,255] using raw_range = [0,2.8]
%   * SAR: clip to [-0.5,0.5] then map to [0,255]
% - If cfg.useZscore is true, per-band z-score is applied AFTER the above.
% - Augmentations are geometric and aligned across bands (spectrally safe).
%
% Matteo Rambaldi — Thesis utilities

function [dsTrain, dsCal, dsTest, info] = DatasetReading(cfg)

arguments
    cfg struct
end

rng(cfgArg(cfg,'randomSeed',42),'twister');

% ---------- 0) Basic checks ----------
trainT = cfgArg(cfg,'trainTable',[]);
testT  = cfgArg(cfg,'testTable',[]);
assert(~isempty(trainT) && ~isempty(testT), 'trainTable and testTable are required.');

% normalize label type
if ~iscategorical(trainT.Label); trainT.Label = categorical(trainT.Label); end
if ~iscategorical(testT.Label);  testT.Label  = categorical(testT.Label);  end
classes = categories(trainT.Label);
numClasses = numel(classes);

% ---------- 1) Split out CALIBRATION by CITY from TRAIN ----------
allCities = unique(string(trainT.CityID));
calFrac   = cfgArg(cfg,'calibrationFrac',0.08);
numCal    = max(1, round(calFrac * numel(allCities)));
calCities  = randsample(allCities, numCal);
trainCities = setdiff(allCities, calCities);

isCalCity  = ismember(string(trainT.CityID), calCities);
calT   = trainT(isCalCity, :);
trainT = trainT(~isCalCity, :);

% ---------- 2) Create raw datastores (lazy I/O) ----------
rd = cfgArg(cfg,'reader',struct('type','auto'));
rd.type      = cfgArg(rd,'type','auto');
rd.field     = cfgArg(rd,'field','patch');
rd.customFcn = cfgArg(rd,'customFcn',[]);

rawTrain = tableDatastore(trainT, rd);
rawCal   = tableDatastore(calT,   rd);
rawTest  = tableDatastore(testT,  rd);

% ---------- 3) Compute per-band stats on TRAIN for optional z-score ----------
fprintf('[DatasetReading] Scanning TRAIN to obtain per-band stats...\n');
[mu, sigma, bands] = computeBandStats(rawTrain, cfgArg(cfg,'inputSize',[32 32]));
fprintf('  -> Bands: %d | mean/std computed over TRAIN.\n', numel(mu));

% ---------- 4) Define preprocessing transforms ----------
pp.msRange   = [0 2.8];
pp.sarClip   = 0.5;
pp.useZscore = cfgArg(cfg,'useZscore',false);
pp.mu        = mu;        % per-band
pp.sigma     = sigma;     % per-band
pp.useSARdespeckle = cfgArg(cfg,'useSARdespeckle',false);
pp.inSize    = cfgArg(cfg,'inputSize',[32 32]);

preproc = @(data) preprocessSample(data, pp);

% ---------- 5) Augmentations (train only; aligned & mild) ----------
doAug = cfgArg(cfg,'useAugmentation',true);
aug = @(data) data; % identity
if doAug
    aug = @(data) augmentAligned(data);
end

% ---------- 6) Wrap into transformed datastores ----------
dsTrain = transform(rawTrain, @(data) aug(preproc(data)));
dsCal   = transform(rawCal,   preproc);
dsTest  = transform(rawTest,  preproc);

% ---------- 7) Class weights (inverse frequency on TRAIN) ----------
cw = classWeightsFromTable(trainT.Label, classes);

% ---------- 8) Pack info ----------
info = struct();
info.numClasses   = numClasses;
info.classes      = classes;
info.bands        = bands;
info.mu           = mu;
info.sigma        = sigma;
info.classWeights = cw;
info.trainCities  = trainCities;
info.calCities    = calCities;
info.testCities   = unique(string(testT.CityID));
info.preprocessFcn= preproc;
info.augmentFcn   = aug;
info.reader       = rd;

fprintf('[DatasetReading] Done. TRAIN %d | CAL %d | TEST %d samples.\n', ...
        height(trainT), height(calT), height(testT));
end

% ======================================================================
%                           Helper functions
% ======================================================================

function v = cfgArg(cfg, name, default)
if isfield(cfg,name), v = cfg.(name); else, v = default; end
end

function ds = tableDatastore(T, rd)
% Returns a datastore that yields a struct with fields:
%   .X (HxWxB single), .Label (categorical), .CityID (string), .Modality ('MS'|'SAR')
readFcn = makeReader(rd);
tt = T; % local copy for nested function
ds = fileDatastore(tt.Path, 'ReadFcn', @(p) packExample(p, tt, readFcn), 'IncludeSubfolders', false);
    function s = packExample(filepath, tab, rf)
        i = find(strcmp(tab.Path, filepath), 1, 'first');
        X = rf(filepath);
        s = struct();
        s.X = X;
        s.Label = tab.Label(i);
        s.CityID = string(tab.CityID(i));
        if ismember('Modality', tab.Properties.VariableNames)
            s.Modality = string(tab.Modality(i));
        else
            s.Modality = "MS"; % default
        end
    end
end

function rf = makeReader(rd)
switch lower(rd.type)
    case 'mat'
        varName = rd.field;
        rf = @(p) single(load(p, varName).(varName));
    case 'tif'
        rf = @(p) single(tiffReadBands(p));
    case 'custom'
        assert(~isempty(rd.customFcn), 'reader.customFcn must be provided for type=custom');
        rf = @(p) single(rd.customFcn(p));
    otherwise % 'auto'
        rf = @(p) autoReader(p, rd);
end
end

function X = autoReader(p, rd)
[~,~,ext] = fileparts(p);
switch lower(ext)
    case '.mat'
        varName = rd.field;
        X = single(load(p, varName).(varName));
    case {'.tif','.tiff'}
        X = single(tiffReadBands(p));
    otherwise
        error('Unsupported file type: %s', ext);
end
end

function A = tiffReadBands(p)
t = Tiff(p,'r');
A = [];
try
    k = 1;
    while true
        setDirectory(t,k);
        img = single(read(t));
        if isempty(A)
            A = zeros([size(img) 1],'single');
        end
        A(:,:,k) = img;
        k = k + 1;
    end
catch
    % reached last directory
end
close(t);
end

function [mu, sigma, bands] = computeBandStats(ds, inSize)
% One pass to estimate per-band mean/std on TRAIN (after paper scaling).
reset(ds);
accMu = []; accSq = []; N = 0; bands = [];
while hasdata(ds)
    s = read(ds);
    X = s.X; % raw HxWxB
    B = size(X,3);
    if isempty(bands), bands = B; end
    Xs = applyPaperScaling(X, guessModality(s));
    Xs = imresize(Xs, inSize, 'nearest');
    x = reshape(Xs, [], B);   % (H*W) x B
    if isempty(accMu)
        accMu = zeros(1,B,'double');
        accSq = zeros(1,B,'double');
    end
    accMu = accMu + sum(x,1);
    accSq = accSq + sum(x.^2,1);
    N = N + size(x,1);
end
mu = single(accMu./N);
sigma = single( sqrt( max(accSq./N - mu.^2, 1e-12) ) );
reset(ds);
end

function s = guessModality(sample)
% sample may be struct or table row; try to read Modality
if isstruct(sample) && isfield(sample,'Modality')
    s = string(sample.Modality);
else
    s = "MS";
end
end

function sample = preprocessSample(sample, pp)
% Apply professor's scaling, optional despeckle (SAR), optional z-score, resize.
X = sample.X;  % HxWxB
modality = guessModality(sample);

% 1) Paper scaling
X = applyPaperScaling(X, modality);

% 2) Optional SAR despeckle (mild)
if pp.useSARdespeckle && modality=="SAR"
    for b=1:size(X,3)
        X(:,:,b) = imnlmfilt(X(:,:,b),'DegreeOfSmoothing',12);
    end
end

% 3) Optional z-score per band (using TRAIN stats)
if pp.useZscore
    mu = reshape(pp.mu, 1,1,[]);
    sg = reshape(pp.sigma, 1,1,[]);
    X = (X - mu) ./ (sg + 1e-6);
end

% 4) Resize to input size (nearest to preserve band structure)
X = imresize(X, pp.inSize, 'nearest');

% Pack back (keep metadata)
sample.X = X; % single
end

function X = applyPaperScaling(X, modality)
% Keep professor's normalization rules, return single.
X = single(X);
if modality == "SAR"
    % clip [-0.5, 0.5] then map to [0,255]
    X = max(min(X, 0.5), -0.5);
    X = (X + 0.5) * 255.0;
else
    % Sentinel-2 MS: raw [0,2.8] -> [0,255]
    X = X ./ (2.8/255.0);
end
end

function out = augmentAligned(sample)
% Aligned geometric augmentations (spectrally safe)
X = sample.X;
sz = size(X);
tform = randomAffine2d( ...
    'XReflection',true, ...
    'Rotation',[-8 8], ...
    'XTranslation',[-2 2], ...
    'YTranslation',[-2 2]);
rout = imref2d(sz(1:2));
for b = 1:sz(3)
    X(:,:,b) = imwarp(X(:,:,b), tform, 'OutputView', rout, 'FillValues', 0);
end
% Optional light cutout (random erasing)
if rand < 0.3
    k = randi([3 5]); % patch side
    [H,W,~]=size(X); x = randi([1,max(1,W-k+1)]); y = randi([1,max(1,H-k+1)]);
    X(y:y+k-1, x:x+k-1, :) = 0;
end
sample.X = X;
out = sample;
end

function cw = classWeightsFromTable(lbl, classes)
% inverse frequency weights (normalized to mean=1)
count = countcats(categorical(lbl, classes));
w = sum(count) ./ max(count,1);
cw = double(w / mean(w));
end