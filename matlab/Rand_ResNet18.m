% Rand_ResNet18
% Sentinel-2 (MS) DenseNet201 ensemble with 3 random bands per member,
% compatible with DatasetReading(cfg) outputs.
%
% Required cfgT fields: dsTrain, dsTest, info
% Optional: numMembers(10), maxEpochs(10), miniBatchSize(32), learnRate(1e-3),
%           rngSeed(1337), plots("none"|"training-progress")
%
% Author: Matteo Rambaldi — Thesis utilities

function res = Rand_ResNet18(cfgT)
    % RAND_DENSENET
    % ResNet18 ensemble trained on Sentinel-2 (MS) with 3 random bands per member.

    % ---------- config ----------
    dsTrain = must(cfgT, 'dsTrain');
    dsTest  = must(cfgT, 'dsTest');
    info    = must(cfgT, 'info');

    numMembers    = getf(cfgT, 'numMembers', 10);
    maxEpochs     = getf(cfgT, 'maxEpochs', 10);
    miniBatchSize = getf(cfgT, 'miniBatchSize', 32);
    learnRate     = getf(cfgT, 'learnRate', 1e-3);
    rngSeed       = getf(cfgT, 'rngSeed', 1337);

    classes    = string(info.classes);
    numClasses = numel(classes);

    % ---------- model ----------
    % Load DenseNet201 from .mat file (pretrained or untrained)
    matfile = fullfile('matlab', 'resnet18_pretrained.mat');
    if ~exist(matfile, 'file')
        error('Missing ''%s''. Please create it locally and upload to HPC.', matfile);
    end
    load(matfile, 'net');
    
    % Convert to layer graph
    lgraph = layerGraph(net);
    inSz   = lgraph.Layers(1).InputSize;

    % Replace classification head for your classes
    % Check the name of the last FC layer in lgraph.Layers first (often 'fc1000')
    lgraph = replaceLayer(lgraph, 'fc1000', ...
        fullyConnectedLayer(numClasses, 'Name','fc', ...
        'WeightLearnRateFactor',20, 'BiasLearnRateFactor',20));
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', classificationLayer('Name','classoutput'));

    % ---------- training options ----------
    opts = trainingOptions('sgdm', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', learnRate, ...
        'Shuffle', 'every-epoch', ...
        'ExecutionEnvironment', 'auto', ...
        'Verbose', false);

    rng(rngSeed, 'twister');

    % ---------- train ensemble ----------
    members(numMembers) = struct('net',[],'bands',[],'valAcc',NaN);
    for m = 1:numMembers
        fprintf('[RAND] Member %02d/%02d\n', m, numMembers);
        bands = randperm(10, 3);  % Random 3 MS bands

        dsTrM = toNetTableMS(dsTrain, bands, inSz(1:2));
        netM  = trainNetwork(dsTrM, lgraph, opts);

        Ytr = classify(netM, dsTrM, 'MiniBatchSize', miniBatchSize);
        Ttr = gatherResponses(dsTrM);
        acc = mean(Ytr == Ttr);

        members(m).net    = netM;
        members(m).bands  = bands;
        members(m).trainAcc = acc;

        fprintf('  bands = [%d %d %d], train acc = %.4f\n', bands, acc);
    end

    % ---------- test ensemble ----------
    fprintf('[RAND] Evaluating ensemble on TEST...\n');
    scoresList = cell(numMembers,1);
    for m = 1:numMembers
        dsTeM = toNetTableMS(dsTest, members(m).bands, inSz(1:2));
        [~, S] = classify(members(m).net, dsTeM, 'MiniBatchSize', miniBatchSize);
        scoresList{m} = S;  % N x K
    end

    Savg = mean(cat(3, scoresList{:}), 3);  % N x K
    [~, idx] = max(Savg, [], 2);

    yTrue = gatherResponses(toNetTableMS(dsTest, members(1).bands, inSz(1:2)));
    yPred = categorical(idx, 1:numClasses, categories(yTrue));

    top1 = mean(yPred == yTrue);
    C = confusionmat(yTrue, yPred, 'Order', categorical(classes));
    fprintf('  Test Top-1 = %.4f\n', top1);

    % ---------- pack results ----------
    res = struct();
    res.members      = members;
    res.testTop1     = top1;
    res.confusionMat = C;
    res.classes      = classes;
    res.scoresAvg    = Savg;
    res.yTrue        = yTrue;
    res.yPred        = yPred;
end

% ----------------- helpers -----------------
function v = must(S,f), assert(isfield(S,f), 'Missing cfgT.%s', f); v=S.(f); end
function v = getf(S,f,d), if isfield(S,f), v=S.(f); else, v=d; end, end

function dsOut = toNetTableMS(dsIn, bands3, hw)
    assert(numel(bands3)==3, 'bands3 must have 3 elements');
    fn = @(s) toRowMS(s, bands3, hw);
    dsOut = transform(dsIn, fn);
end

function T = toRowMS(sample, bands3, hw)
    X = sample.X(:,:,bands3);
    if size(X,1)~=hw(1) || size(X,2)~=hw(2)
        X = imresize(X, hw, 'bilinear');
    end
    T = table({X}, categorical(sample.Label), 'VariableNames', {'input','response'});
end

function Y = gatherResponses(dsTbl)
    reset(dsTbl);
    Y = categorical.empty(0,1);
    while hasdata(dsTbl)
        t = read(dsTbl);
        Y(end+1,1) = t.response; %#ok<AGROW>
    end
    reset(dsTbl);
end
