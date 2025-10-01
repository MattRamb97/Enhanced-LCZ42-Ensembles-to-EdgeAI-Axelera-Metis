% run_fusion.m — fuse Rand, RandRGB, SAR teachers with optional calibration
rng(42,'twister');

addpath(genpath('matlab'));

% 1) Load trained teacher results (must contain: scoresAvg, yTrue, classes)
load matlab/resRand.mat      % -> resRand
load matlab/resRandRGB.mat   % -> resRandRGB
load matlab/resSAR.mat       % -> resSAR

% 2) If calibration is missing in any model, compute it now
needCal = @(s) ~(isfield(s,'scoresCal') && isfield(s,'yCal'));

if needCal(resRand) || needCal(resRandRGB) || needCal(resSAR)
    % 2a) Rebuild CAL/TEST datastores for MS and SAR (same order as training)
    dataRoot = 'data/lcz42';

    % MS tables
    S_MS = load(fullfile(dataRoot,'tables_MS.mat'));   % train_MS, val_MS, test_MS
    cfgMS = struct;
    cfgMS.trainTable = S_MS.train_MS;
    cfgMS.testTable  = S_MS.test_MS;
    cfgMS.useZscore = true;
    cfgMS.useSARdespeckle = true;
    cfgMS.useAugmentation = false;       % no aug for eval
    cfgMS.reader = struct('type','custom','customFcn',@(row) h5_reader(row.Path,row.Index,row.Modality));
    [~, dsCal_MS, dsTest_MS, infoMS] = DatasetReading(cfgMS); %#ok<ASGLU>

    % SAR tables
    S_SAR = load(fullfile(dataRoot,'tables_SAR.mat')); % train_SAR, val_SAR, test_SAR
    cfgSAR = struct;
    cfgSAR.trainTable = S_SAR.train_SAR;
    cfgSAR.testTable  = S_SAR.test_SAR;
    cfgSAR.useZscore = true;
    cfgSAR.useSARdespeckle = true;
    cfgSAR.useAugmentation = false;
    cfgSAR.reader = struct('type','custom','customFcn',@(row) h5_reader(row.Path,row.Index,row.Modality));
    [~, dsCal_SAR, dsTest_SAR, infoSAR] = DatasetReading(cfgSAR); %#ok<ASGLU>

    % 2b) Compute calibration probs for each model if missing
    % MS backbones use member-specific 3-band views
    if needCal(resRand)
        fprintf('[CAL] Rand_DenseNet (MS random 3-band)…\n');
        [resRand.scoresCal, resRand.yCal] = avg_softmax_on_MS(resRand.members, dsCal_MS);
        resRand.name = "Rand";
    end
    if needCal(resRandRGB)
        fprintf('[CAL] RandRGB_DenseNet (MS ≥1 RGB)…\n');
        [resRandRGB.scoresCal, resRandRGB.yCal] = avg_softmax_on_MS(resRandRGB.members, dsCal_MS);
        resRandRGB.name = "RandRGB";
    end
    % SAR backbone uses engineered triplet: [VV_lee (6), VH_lee (5), |C12| (7,8)]
    if needCal(resSAR)
        fprintf('[CAL] SAR DenseNet (engineered triplet)…\n');
        [resSAR.scoresCal, resSAR.yCal] = avg_softmax_on_SAR_engineered(resSAR.members, dsCal_SAR);
        resSAR.name = "SAR";
    end
end

% 3) Ensure model names (nice to have)
if ~isfield(resRand,'name'),     resRand.name = "Rand";     end
if ~isfield(resRandRGB,'name'),  resRandRGB.name = "RandRGB"; end
if ~isfield(resSAR,'name'),      resSAR.name = "SAR";       end

% 4) Fuse
fusion = Fusion_Ensemble(struct('models',{resRand, resRandRGB, resSAR}));

% 5) Save fusion results
save('matlab/fusionResults.mat','fusion','-v7.3');

fprintf('\n[Fusion] Done. Accuracies →  SUM: %.4f  |  TEMP: %s  |  STACK: %s\n', ...
    fusion.sum.acc, ...
    ternary(hasfield(fusion,'temp') && ~isempty(fusion.temp), sprintf('%.4f',fusion.temp.acc), 'n/a'), ...
    ternary(hasfield(fusion,'stack')&& ~isempty(fusion.stack), sprintf('%.4f',fusion.stack.acc),'n/a'));

% ======================= local helpers =======================

function tf = hasfield(s,f), tf = isstruct(s) && isfield(s,f); end
function s = ternary(cond,a,b), if cond, s=a; else, s=b; end, end

function [Pavg, y] = avg_softmax_on_MS(members, dsMS)
% Average softmax across members for MS models; each member has .bands (1..10)
reset(dsMS);
% Build per-member table datastores matching their 3-band selection
inSz = [224 224];              % DenseNet-201 input spatial size (channels=3)
M = numel(members);
dsM = cell(M,1);
for m = 1:M
    dsM{m} = transform(dsMS, @(s) rowMS_3bands(s, members(m).bands, inSz));
end
% Read once through and accumulate predictions
P = []; y = [];
while hasdata(dsM{1})
    t = cellfun(@(d) read(d), dsM, 'UniformOutput', false);
    % All t{m} share the same response in this row
    y(end+1,1) = t{1}.response; %#ok<AGROW>
    % Sum softmax across members
    pSum = 0;
    for m = 1:M
        net = members(m).net;
        [~, p] = classify(net, t{m}(:,1), 'MiniBatchSize', 64);
        pSum = pSum + single(p);
    end
    P(end+1,1:size(pSum,2)) = pSum ./ M; %#ok<AGROW>
end
reset(dsMS);
Pavg = single(P);
y    = categorical(y);
end

function row = rowMS_3bands(sample, bands3, hw)
X = sample.X;        % [H W 10] single [0,255]
assert(numel(bands3)==3 && size(X,3)>=max(bands3));
X3 = X(:,:,bands3);
if size(X3,1)~=hw(1) || size(X3,2)~=hw(2)
    X3 = imresize(X3, hw, 'bilinear');
end
row = table({X3}, categorical(sample.Label), 'VariableNames', {'input','response'});
end

function [Pavg, y] = avg_softmax_on_SAR_engineered(members, dsSAR)
% Average softmax for SAR engineered triplet [VV_lee(6), VH_lee(5), |C12|]
reset(dsSAR);
inSz = [224 224];
M = numel(members);
dsM = cell(M,1);
tripletFcn = @(X) cat(3, X(:,:,6), X(:,:,5), hypot(X(:,:,7), X(:,:,8)));
for m = 1:M
    dsM{m} = transform(dsSAR, @(s) rowSAR_engineered(s, tripletFcn, inSz));
end
P = []; y = [];
while hasdata(dsM{1})
    t = cellfun(@(d) read(d), dsM, 'UniformOutput', false);
    y(end+1,1) = t{1}.response; %#ok<AGROW>
    pSum = 0;
    for m = 1:M
        net = members(m).net;
        [~, p] = classify(net, t{m}(:,1), 'MiniBatchSize', 64);
        pSum = pSum + single(p);
    end
    P(end+1,1:size(pSum,2)) = pSum ./ M; %#ok<AGROW>
end
reset(dsSAR);
Pavg = single(P);
y    = categorical(y);
end

function row = rowSAR_engineered(sample, ftrip, hw)
X = sample.X;        % [H W 8] single [0,255]
X3 = ftrip(X);
if size(X3,1)~=hw(1) || size(X3,2)~=hw(2)
    X3 = imresize(X3, hw, 'bilinear');
end
row = table({X3}, categorical(sample.Label), 'VariableNames', {'input','response'});
end