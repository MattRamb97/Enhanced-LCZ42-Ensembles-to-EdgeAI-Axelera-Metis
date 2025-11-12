% download the dataset from
% https://dataserv.ub.tum.de/index.php/s/m1483140

% DATASETREADING  Build TRAIN/TEST datastores for So2Sat LCZ42 with HDF5 backing.
%
% Required:
%   cfg.trainTable : table with {'Path','Label','Index','Modality'}
%   cfg.testTable  : table with {'Path','Label','Index','Modality'}
%
% Optional:
%   cfg.inputSize        (default [32 32])
%   cfg.useZscore        (default false)      % compute μ/σ on TRAIN after paper scaling
%   cfg.useSARdespeckle  (default false)      % mild optional filtering
%   cfg.useAugmentation  (default false)      % baseline-first; enable later
%   cfg.calibrationFrac  (default 0.08)       % class-stratified CAL split from TRAIN
%   cfg.randomSeed       (default 42)
%   cfg.reader.type      ('custom'|'mat'|'tif'|'auto', default 'custom')
%   cfg.reader.field     (for .mat, default 'patch')
%   cfg.reader.customFcn (@(row)-> HxWxC single)  % e.g., h5_reader(row.Path,row.Index,row.Modality)
%
% Output datastores yield structs:
%   .X (HxWxC single), .Label (categorical), .Modality (string)
%
% Paper scaling (faithful to baseline):
%   MS  (/sen2): values in [0,2.8] -> scale to [0,255]
%   SAR (/sen1): clip to [-0.5,0.5] -> shift/scale to [0,255]
%
% NOTES
% - This loader preserves the professor's normalization:
%   * Multispectral (MS): scale to [0,255] using raw_range = [0,2.8]
%   * SAR: clip to [-0.5,0.5] then map to [0,255]
% - If cfg.useZscore is true, per-band z-score is applied AFTER the above.
% - Augmentations are geometric and aligned across bands (spectrally safe).
%
% Matteo Rambaldi — Thesis utilities

function [dsTrain, dsTest, info] = DatasetReading(cfg)

    arguments
        cfg struct
    end
    
    % ------------------ 0) Args and checks ------------------
    rng(cfgArg(cfg,'randomSeed',42),'twister');
    
    trainT = cfgArg(cfg,'trainTable',[]);
    testT  = cfgArg(cfg,'testTable',[]);
    assert(~isempty(trainT) && ~isempty(testT), 'cfg.trainTable and cfg.testTable are required.');

    assert(all(ismember({'Path','Label','Index','Modality'}, trainT.Properties.VariableNames)), ...
           'trainTable must have Path, Label, Index, Modality.');
    assert(all(ismember({'Path','Label','Index','Modality'}, testT.Properties.VariableNames)), ...
           'testTable must have Path, Label, Index, Modality.');
    
    % Normalize labels to categorical
    if ~iscategorical(trainT.Label), trainT.Label = categorical(trainT.Label); end
    if ~iscategorical(testT.Label),  testT.Label  = categorical(testT.Label);  end
    classes    = categories(trainT.Label);
    numClasses = numel(classes);
    
    inSize   = cfgArg(cfg,'inputSize',[32 32]);
    useZS    = cfgArg(cfg,'useZscore',false);
    useSARdn = cfgArg(cfg,'useSARdespeckle',false);
    useAug   = cfgArg(cfg,'useAugmentation',false);   % baseline-first
    
    rd = cfgArg(cfg,'reader',struct('type','custom'));
    rd.type      = cfgArg(rd,'type','custom');
    rd.customFcn = cfgArg(rd,'customFcn',[]);
    
    % ------------------ 1) Row-wise datastores via arrayDatastore+transform ------------------
    rawTrain = tableDatastore(trainT, rd);
    rawTest  = tableDatastore(testT,  rd);
    
    % ------------------ 2) Per-channel μ/σ on TRAIN (after paper scaling) ------------------
    fprintf('[DatasetReading] Computing per-channel μ/σ on TRAIN...\n');
    [mu, sigma, numCh] = computeBandStats(rawTrain, inSize);
    fprintf('  -> Channels: %d | μ/σ computed.\n', numCh);
    
    % ------------------ 3) Preprocessing pipeline ------------------
    pp = struct();
    pp.msRange   = [0 2.8];
    pp.sarClip   = 0.5;
    pp.useZscore = useZS;
    pp.mu        = mu;
    pp.sigma     = sigma;
    pp.useSARdespeckle = useSARdn;
    pp.inSize    = inSize;
    
    preproc = @(s) preprocessSample(s, pp);
    
    % Augmentations (aligned, mild) only on TRAIN
    aug = @(s) s;
    if useAug, aug = @(s) augmentAligned(s); end
    
    dsTrain = transform(rawTrain, @(s) aug(preproc(s)));
    dsTest  = transform(rawTest,  preproc);
    
    % ------------------ 4) Info ------------------
    info = struct();
    info.numClasses   = numClasses;
    info.classes      = classes;
    info.mu           = mu;
    info.sigma        = sigma;
    info.inputSize    = inSize;
    
    fprintf('[DatasetReading] Done. TRAIN %d | TEST %d.\n', ...
            height(trainT), height(testT));
end

% ======================================================================
%                               HELPERS
% ======================================================================

function v = cfgArg(cfg, name, default)
    if isfield(cfg,name), v = cfg.(name); else, v = default; end
end

function ds = tableDatastore(T, rd)
    % Datastore over table rows -> each element returns a struct {X,Label,Modality}
    rowFn = @(idx) packByIndex(T, rd, idx);
    
    ids = (1:height(T))';                      % numeric indices
    ads = arrayDatastore(ids, 'ReadSize', 1);  % one index per read
    ds  = transform(ads, @(i) rowFn(i));       % i is numeric scalar in R2025b
end

function s = packByIndex(T, rd, idx)
    % idx can be numeric or (rarely) a 1x1 cell; unwrap robustly
    while iscell(idx), idx = idx{1}; end
    validateattributes(idx, {'numeric'}, {'scalar','integer','>=',1,'<=',height(T)});
    row = T(idx, :);
    
    switch lower(rd.type)
        case 'custom'
            assert(~isempty(rd.customFcn), 'cfg.reader.customFcn must be set for type="custom".');
            X = rd.customFcn(row);                 % must return HxWxC single
        case 'mat'
            varName = rd.field;
            S = load(row.Path, varName);
            X = single(S.(varName));
        case 'tif'
            X = single(imread(row.Path));
        otherwise % 'auto'
            [~,~,ext] = fileparts(row.Path);
            switch lower(ext)
                case '.mat'
                    varName = rd.field;
                    S = load(row.Path, varName);
                    X = single(S.(varName));
                case {'.tif','.tiff','.png','.jpg','.jpeg'}
                    X = single(imread(row.Path));
                otherwise
                    error('AUTO reader: unsupported extension %s', ext);
            end
    end
    
    % Package
    s = struct('X',X,'Label',row.Label,'Modality',string(row.Modality));
end

function [mu, sigma, C] = computeBandStats(ds, inSize)
    reset(ds);
    accMu = []; accSq = []; Npix = 0; C = [];
    while hasdata(ds)
        s = read(ds);
        X = applyPaperScaling(s.X, s.Modality);
        X = imresize(X, inSize, 'nearest');   % keep bands aligned
        C = size(X,3);
        x = reshape(single(X), [], C);        % (H*W) x C
        if isempty(accMu), accMu = zeros(1,C,'double'); accSq = zeros(1,C,'double'); end
        accMu = accMu + sum(x,1);
        accSq = accSq + sum(x.^2,1);
        Npix = Npix + size(x,1);
    end
    mu    = single(accMu ./ Npix);
    sigma = single( sqrt( max(accSq./Npix - mu.^2, 1e-12) ) );
    reset(ds);
end

function sample = preprocessSample(sample, pp)
    X = sample.X;                      % HxWxC single
    modality = string(sample.Modality);
    
    % 1) Paper scaling (faithful to baseline)
    X = applyPaperScaling(X, modality);
    
    % 2) Optional SAR despeckle (very mild)
    if pp.useSARdespeckle && modality=="SAR"
        for c=1:size(X,3)
            X(:,:,c) = imnlmfilt(X(:,:,c),'DegreeOfSmoothing',12);
        end
    end
    
    % 3) Optional z-score (per channel)
    if pp.useZscore
        mu = reshape(pp.mu, 1,1,[]);
        sg = reshape(pp.sigma, 1,1,[]);
        X = (X - mu) ./ (sg + 1e-6);
    end
    
    % 4) Resize to network input
    X = imresize(X, pp.inSize, 'nearest');
    
    sample.X = X;
end

function X = applyPaperScaling(X, modality)
    X = single(X);
    if modality=="SAR"
        % clip [-0.5,0.5] -> shift/scale to [0,255]
        X = max(min(X, 0.5), -0.5);
        X = (X + 0.5) * 255.0;
    else
        % [0,2.8] -> [0,255]
        X = X ./ (2.8/255.0);
    end
end

function out = augmentAligned(sample)
    X = sample.X;
    sz = size(X);
    tform = randomAffine2d( ...
        'XReflection',true, 'Rotation',[-8 8], ...
        'XTranslation',[-2 2], 'YTranslation',[-2 2]);
    rout = imref2d(sz(1:2));
    for c=1:sz(3)
        X(:,:,c) = imwarp(X(:,:,c), tform, 'OutputView', rout, 'FillValues', 0);
    end
    % light cutout
    if rand < 0.3
        k = randi([3 5]); [H,W,~]=size(X);
        x = randi([1,max(1,W-k+1)]); y = randi([1,max(1,H-k+1)]);
        X(y:y+k-1, x:x+k-1, :) = 0;
    end
    sample.X = X;
    out = sample;
end