% ENSEMBLESARCHANNEL_DENSENET
% SAR (SEN1) DenseNet201 ensemble with 3 random bands per member,
% compatible with DatasetReading(cfg) outputs.
%
% Required fields in cfgT:
%   dsTrain, dsCal, dsTest, info
% Optional:
%   numMembers (default 10)   % professor used 10
%   maxEpochs  (default 7)    % professor used 7
%   miniBatchSize (default 50)
%   learnRate  (default 1e-3)
%   rngSeed    (default 1337)
%   plots      (default "none") % or "training-progress"
%
% Author: Matteo Rambaldi — Thesis utilities

function res = ensembleSARchannel_DenseNet(cfgT)
    % Ensemble of DenseNet201 models trained on 3 random SAR channels

    % ---------- config ----------
    dsTrain = must(cfgT, 'dsTrain');
    dsTest  = must(cfgT, 'dsTest');
    info    = must(cfgT, 'info');

    numMembers    = getf(cfgT, 'numMembers', 10);
    maxEpochs     = getf(cfgT, 'maxEpochs', 12);
    miniBatchSize = getf(cfgT, 'miniBatchSize', 128);
    learnRate     = getf(cfgT, 'learnRate', 1e-3);
    rngSeed       = getf(cfgT, 'rngSeed', 1337);

    classes    = string(info.classes);
    numClasses = numel(classes);

    % ---------- base network ----------
    lgraph = densenet201('Weights','none');
    inSz   = lgraph.Layers(1).InputSize;

    lgraph = replaceLayer(lgraph,'fc1000', ...
        fullyConnectedLayer(numClasses,'Name','fc', ...
            'WeightLearnRateFactor',20,'BiasLearnRateFactor',20));
    lgraph = replaceLayer(lgraph,'fc1000_softmax', softmaxLayer('Name','softmax'));
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000', classificationLayer('Name','classoutput'));

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
    members(numMembers) = struct('net',[],'bands',[],'trainAcc',NaN);
    for m = 1:numMembers
        fprintf('[SAR] Member %02d/%02d\n', m, numMembers);
        bands = randperm(8,3);  % 3 random SAR bands

        dsTrM = toNetTableSAR(dsTrain, bands, inSz(1:2));
        netM  = trainNetwork(dsTrM, lgraph, opts);

        Ytr = classify(netM, dsTrM, 'MiniBatchSize', miniBatchSize);
        Ttr = gatherResponses(dsTrM);
        acc = mean(Ytr == Ttr);

        members(m).net   = netM;
        members(m).bands = bands;
        members(m).trainAcc = acc;

        fprintf('  bands = [%d %d %d], train acc = %.4f\n', bands, acc);
    end

    % ---------- test ensemble ----------
    fprintf('[SAR] Evaluating ensemble on TEST...\n');
    scoresList = cell(numMembers,1);
    for m = 1:numMembers
        dsTeM = toNetTableSAR(dsTest, members(m).bands, inSz(1:2));
        [~, S] = classify(members(m).net, dsTeM, 'MiniBatchSize', miniBatchSize);
        scoresList{m} = S;
    end

    Savg = mean(cat(3, scoresList{:}), 3);  % N x K
    [~, idx] = max(Savg, [], 2);

    yTrue = gatherResponses(toNetTableSAR(dsTest, members(1).bands, inSz(1:2)));
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

% ------------------ helpers ------------------

function v = must(S,f), assert(isfield(S,f),'Missing cfgT.%s',f); v=S.(f); end
function v = getf(S,f,d), if isfield(S,f), v=S.(f); else, v=d; end, end

function dsOut = toNetTableSAR(dsIn, bands, hw)
    fn = @(s) toRowSAR(s, bands, hw);
    dsOut = transform(dsIn, fn);
end

function T = toRowSAR(sample, bands, hw)
    X = sample.X;
    assert(size(X,3) >= max(bands), 'Sample has %d channels, asked for band %d.', size(X,3), max(bands));
    X3 = X(:,:,bands);
    if size(X3,1) ~= hw(1) || size(X3,2) ~= hw(2)
        X3 = imresize(X3, hw, 'bilinear');
    end
    T = table({X3}, categorical(sample.Label), 'VariableNames', {'input','response'});
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