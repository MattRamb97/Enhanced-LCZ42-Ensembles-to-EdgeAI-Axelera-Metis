% RAND_DENSENET
% Sentinel-2 (MS) DenseNet201 ensemble with 3 random bands per member,
% compatible with DatasetReading(cfg) outputs.
%
% Required cfgT fields: dsTrain, dsCal, dsTest, info
% Optional: numMembers(10), maxEpochs(7), miniBatchSize(50), learnRate(1e-3),
%           rngSeed(1337), plots("none"|"training-progress")
%
% Matteo Rambaldi — Thesis utilities

function res = Rand_DenseNet(cfgT)
    
    % ---------- config ----------
    dsTrain = must(cfgT,'dsTrain');
    dsCal   = must(cfgT,'dsCal');
    dsTest  = must(cfgT,'dsTest');
    info    = must(cfgT,'info');
    
    numMembers    = getf(cfgT,'numMembers',10);
    maxEpochs     = getf(cfgT,'maxEpochs',7);
    miniBatchSize = getf(cfgT,'miniBatchSize',50);
    learnRate     = getf(cfgT,'learnRate',1e-3);
    rngSeed       = getf(cfgT,'rngSeed',1337);
    plotsOpt      = string(getf(cfgT,'plots',"none"));
    
    classes    = string(info.classes);
    numClasses = numel(classes);
    
    % ---------- backbone & head ----------
    netBase = densenet201;                    % Deep Learning Toolbox Model for DenseNet
    inSz    = netBase.Layers(1).InputSize;    % e.g., [224 224 3]
    assert(inSz(3)==3,'Network must accept 3 channels.');
    
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
        'ValidationData',[], ...
        'ExecutionEnvironment','auto', ...
        'Verbose',false, ...
        'Plots',plotsOpt);
    
    rng(rngSeed,'twister');
    
    % ---------- train ensemble ----------
    members(numMembers) = struct('net',[],'bands',[],'valAcc',NaN);
    for m = 1:numMembers
        fprintf('[MS] Member %02d/%02d\n', m, numMembers);
    
        % 3 random bands out of 10 (S2: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12)
        bands = randperm(10,3);
    
        % member-specific views: slice chosen 3 bands, resize, yield {input,response}
        dsTrM  = toNetTableMS(dsTrain, bands, inSz(1:2));
        dsCalM = toNetTableMS(dsCal,   bands, inSz(1:2));
    
        % set validation data
        opts.ValidationData    = dsCalM;
        opts.ValidationFrequency = 200;
        opts.ValidationPatience  = 6;
    
        % train
        netM = trainNetwork(dsTrM, lgraph, opts);
    
        % quick val acc
        Yv = classify(netM, dsCalM, 'MiniBatchSize', miniBatchSize);
        Tv = gatherResponses(dsCalM);
        members(m).valAcc = mean(Yv==Tv);
        members(m).net    = netM;
        members(m).bands  = bands;
    
        fprintf('  bands = [%d %d %d], val acc = %.4f\n', bands, members(m).valAcc);
    end
    
    % ---------- evaluate: average softmax across members ----------
    fprintf('[MS] Evaluating ensemble on TEST...\n');
    scoresList = cell(numMembers,1);
    for m = 1:numMembers
        dsTeM = toNetTableMS(dsTest, members(m).bands, inSz(1:2));
        [~, S] = classify(members(m).net, dsTeM, 'MiniBatchSize', miniBatchSize);
        scoresList{m} = S;  % N x K
    end
    
    N = size(scoresList{1},1);
    K = size(scoresList{1},2);
    Ssum = zeros(N,K,'single');
    for m = 1:numMembers, Ssum = Ssum + scoresList{m}; end
    Savg = Ssum / numMembers;
    
    [~, idx] = max(Savg,[],2);
    % any member mapping has same order → use its label stream to build categories
    yTrue = gatherResponses(toNetTableMS(dsTest, members(1).bands, inSz(1:2)));
    yPred = categorical(idx, 1:K, categories(yTrue));
    
    top1 = mean(yPred==yTrue);
    C    = confusionmat(yTrue, yPred, 'Order', categorical(classes));
    fprintf('  Test Top-1 = %.4f\n', top1);
    
    % ---------- pack ----------
    res = struct();
    res.members      = members;
    res.testTop1     = top1;
    res.confusionMat = C;
    res.classes      = classes;
    res.scoresAvg    = Savg;     % N x K
    res.yTrue        = yTrue;
    res.yPred        = yPred;
end

% ================= helpers =================
function v = must(S,f), assert(isfield(S,f),'Missing cfgT.%s',f); v=S.(f); end
function v = getf(S,f,d), if isfield(S,f), v=S.(f); else, v=d; end, end

function dsOut = toNetTableMS(dsIn, bands3, hw)
    assert(numel(bands3)==3,'bands3 must have 3 indices.');
    fn = @(s) toRowMS(s, bands3, hw);
    dsOut = transform(dsIn, fn);
end

function T = toRowMS(sample, bands3, hw)
    % sample.X: [H W 10] single in [0,255] (scaled in DatasetReading)
    X = sample.X;
    assert(size(X,3) >= max(bands3), 'Sample has %d channels, asked for band %d.', size(X,3), max(bands3));
    X3 = X(:,:,bands3);
    if size(X3,1)~=hw(1) || size(X3,2)~=hw(2)
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