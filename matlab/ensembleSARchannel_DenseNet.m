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
% Matteo Rambaldi — Thesis utilities

function res = ensembleSARchannel_DenseNet(cfgT)

    gpuDevice(1);
    
    % ---------- read config ----------
    dsTrain = must(cfgT,'dsTrain');   % TransformedDatastore from DatasetReading
    dsCal   = must(cfgT,'dsCal');
    dsTest  = must(cfgT,'dsTest');
    info    = must(cfgT,'info');
    
    numMembers    = getf(cfgT,'numMembers',10);
    maxEpochs     = getf(cfgT,'maxEpochs',7);
    miniBatchSize = getf(cfgT,'miniBatchSize',50);
    learnRate     = getf(cfgT,'learnRate',1e-3);
    rngSeed       = getf(cfgT,'rngSeed',1337);
    plotsOpt      = string(getf(cfgT,'plots',"none"));
    
    assert(iscategorical(info.classes) || iscell(info.classes) || isstring(info.classes), ...
        'info.classes must be categories/strings.');
    classes    = string(info.classes);
    numClasses = numel(classes);
    
    % ---------- base net & input size ----------
    netBase = densenet201;                     % requires: Deep Learning Toolbox Model for DenseNet
    inSz    = netBase.Layers(1).InputSize;     % e.g., [224 224 3]
    assert(inSz(3)==3, 'Network must accept 3-channel input.');
    
    % Replace head (robust to R2025b)
    lgraph = layerGraph(netBase);
    lgraph = replaceLayer(lgraph,'fc1000',fullyConnectedLayer(numClasses,'Name','fc', ...
                         'WeightLearnRateFactor',20,'BiasLearnRateFactor',20));
    lgraph = replaceLayer(lgraph,'fc1000_softmax',softmaxLayer('Name','softmax'));
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',classificationLayer('Name','classoutput'));
    
    % ---------- training options ----------
    opts = trainingOptions('sgdm', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',learnRate, ...
        'Shuffle','every-epoch', ...
        'ValidationData',[], ...                      % per-member we'll set it
        'ExecutionEnvironment','auto', ...
        'Verbose',false, ...
        'Plots',plotsOpt);
    
    rng(rngSeed,'twister');
    
    % ---------- train each ensemble member ----------
    members(numMembers) = struct('net',[],'bands',[],'valAcc',NaN);
    for m = 1:numMembers
        fprintf('[SAR] Member %02d/%02d\n', m, numMembers);
    
        % choose 3 distinct SAR bands out of 8 (like professor)
        bands = randperm(8,3);
    
        % member-specific transforms: pick 3 bands and resize → 2-col table (input, response)
        dsTrM  = toNetTableSAR(dsTrain, bands, inSz(1:2));
        dsCalM = toNetTableSAR(dsCal,   bands, inSz(1:2));
    
        % set validation data
        opts.ValidationData = dsCalM;
        opts.ValidationFrequency = 200;
        opts.ValidationPatience  = 6;
    
        % train
        netM = trainNetwork(dsTrM, lgraph, opts);
    
        % quick val acc
        Yv = classify(netM, dsCalM, 'MiniBatchSize', miniBatchSize);
        Tv = gatherResponses(dsCalM);
        valAcc = mean(Yv == Tv);
    
        members(m).net   = netM;
        members(m).bands = bands;
        members(m).valAcc = valAcc;
    
        fprintf('  bands = [%d %d %d], val acc = %.4f\n', bands, valAcc);
    end
    
    % ---------- evaluate ensemble (sum rule over softmax scores) ----------
    fprintf('[SAR] Evaluating ensemble on TEST...\n');
    % reuse per-member test transforms to ensure identical channel triplets
    scoresList = cell(numMembers,1);
    for m = 1:numMembers
        dsTeM = toNetTableSAR(dsTest, members(m).bands, inSz(1:2));
        [~, S] = classify(members(m).net, dsTeM, 'MiniBatchSize', miniBatchSize);
        scoresList{m} = S;   % N x K
    end
    
    % all members produce scores for the same samples/order ⇒ average
    N = size(scoresList{1},1);
    K = size(scoresList{1},2);
    Ssum = zeros(N,K,'single');
    for m = 1:numMembers, Ssum = Ssum + scoresList{m}; end
    Savg = Ssum / numMembers;
    
    [~, idx] = max(Savg, [], 2);
    yPred = categorical(idx, 1:K, categories(gatherResponses(toNetTableSAR(dsTest, 1:3, inSz(1:2)))));
    
    % get ground-truth once (any member mapping yields same order)
    yTrue = gatherResponses(toNetTableSAR(dsTest, members(1).bands, inSz(1:2)));
    
    % metrics
    top1 = mean(yPred == yTrue);
    C    = confusionmat(yTrue, yPred, 'Order', categorical(classes));
    
    fprintf('  Test Top-1 = %.4f\n', top1);
    
    % ---------- package results ----------
    res = struct();
    res.members      = members;          % nets + bands + valAcc
    res.testTop1     = top1;
    res.confusionMat = C;
    res.classes      = classes;
    res.scoresAvg    = Savg;             % N x K (softmax-averaged)
    res.yTrue        = yTrue;
    res.yPred        = yPred;
end

% ======================== local helpers ========================

function v = must(S, f)
    assert(isfield(S,f), 'Missing required cfgT field: %s', f);
    v = S.(f);
end

function v = getf(S, f, d)
    if isfield(S,f), v = S.(f); else, v = d; end
end

function dsOut = toNetTableSAR(dsIn, bands3, hw)
    % Transform a DatasetReading datastore (struct with fields X,Label,Modality)
    % into a 2-col table datastore (input, response) for trainNetwork.
    % Picks specified 3 SAR bands and resizes to [hw].
    assert(numel(bands3)==3, 'bands3 must have 3 indices.');
    bands3 = sort(bands3(:).');  % stability not required, but ok
    
    fn = @(s) toRow(s, bands3, hw);
    dsOut = transform(dsIn, fn);
end

function T = toRow(sample, bands3, hw)
    % sample.X: [H W C] single in [0,255]; SAR has C==8
    X = sample.X;
    assert(size(X,3) >= max(bands3), 'Sample has only %d channels, asked for band %d.', size(X,3), max(bands3));
    X3 = X(:,:,bands3);
    
    % Resize to network input
    if size(X3,1) ~= hw(1) || size(X3,2) ~= hw(2)
        X3 = imresize(X3, hw, 'bilinear');
    end
    
    % Return a 2-col table (input, response). One row per read.
    T = table({X3}, categorical(sample.Label), 'VariableNames', {'input','response'});
end

function Y = gatherResponses(dsTbl)
    % dsTbl: datastore returning 1x2 table with 'response' categorical
    reset(dsTbl);
    Y = categorical.empty(0,1);
    while hasdata(dsTbl)
        t = read(dsTbl);
        Y(end+1,1) = t.response; %#ok<AGROW>
    end
    reset(dsTbl);
end