% randSAR_ResNet18
% Mixed MS+SAR ResNet18 ensemble with 2 MS + 1 SAR channels per member,
% compatible with DatasetReading(cfg) outputs.
%
% Required fields in cfgT:
%   dsTrain       : MS datastore (output of DatasetReading)
%   dsTest        : MS datastore
%   dsTrainSAR    : SAR datastore
%   dsTestSAR     : SAR datastore
%   info          : Info struct from DatasetReading (classes, stats)
%
% Optional cfgT fields:
%   numMembers     (default: 10)    % Number of ensemble members
%   maxEpochs      (default: 10)    % Number of training epochs
%   miniBatchSize  (default: 512)   % Mini-batch size
%   learnRate      (default: 1e-3)  % Learning rate
%   rngSeed        (default: 1337)  % Random seed for reproducibility
%   plots          (default: "none") % or "training-progress"
%
% Each ensemble member is trained on a different random combination
% of 2 Sentinel-2 (MS) bands and 1 Sentinel-1 (SAR) band.
%
% Author: Matteo Rambaldi — Thesis utilities

function res = randSAR_ResNet18(cfgT)
    % Ensemble with 2 Multispectral + 1 SAR channel per member
    
    % ---------- config ----------
    dsTrain = must(cfgT, 'dsTrain');  % This should be MS datastore
    dsTest  = must(cfgT, 'dsTest');   % This should be MS datastore
    dsTrainSAR = must(cfgT, 'dsTrainSAR');  % Need SAR datastore too!
    dsTestSAR  = must(cfgT, 'dsTestSAR');   % Need SAR datastore too!
    info    = must(cfgT, 'info');

    numMembers    = getf(cfgT, 'numMembers', 10);
    maxEpochs     = getf(cfgT, 'maxEpochs', 10);
    miniBatchSize = getf(cfgT, 'miniBatchSize', 32);
    learnRate     = getf(cfgT, 'learnRate', 1e-3);
    rngSeed       = getf(cfgT, 'rngSeed', 1337);

    classes    = string(info.classes);
    numClasses = numel(classes);

    % ---------- base network ----------
    matfile = fullfile('matlab', 'resnet18_pretrained.mat');
    if ~exist(matfile, 'file')
        error('Missing ''%s''. Please create it locally and upload to HPC.', matfile);
    end
    
    load(matfile, 'net');
    lgraph = layerGraph(net);
    inSz = lgraph.Layers(1).InputSize;

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
    members(numMembers) = struct('net',[],'msBands',[],'sarBand',[],'trainAcc',NaN);
    
    for m = 1:numMembers
        fprintf('[MS+SAR] Member %02d/%02d\n', m, numMembers);
        
        % Select 2 MS bands and 1 SAR band
        msBands = randperm(10, 2);   % 2 random MS bands from 10
        sarBand = randi(8);          % 1 random SAR band from 8

        % Create mixed datastore for training
        dsTrM = toNetTableMSSAR(dsTrain, dsTrainSAR, msBands, sarBand, inSz(1:2));
        
        % Train network
        netM = trainNetwork(dsTrM, lgraph, opts);
        dlnetM = dlnetwork(netM);

        % Evaluate on training set
        Ytr = classify(netM, dsTrM, 'MiniBatchSize', miniBatchSize);
        Ttr = gatherResponses(dsTrM);
        acc = mean(Ytr == Ttr);

        members(m).net = dlnetM;
        members(m).msBands = msBands;
        members(m).sarBand = sarBand;
        members(m).trainAcc = acc;

        fprintf('  MS bands = [%d %d], SAR band = %d, train acc = %.4f\n', ...
                msBands, sarBand, acc);
    end

    % ---------- test ensemble ----------
    fprintf('[MS+SAR] Evaluating ensemble on TEST...\n');
    scoresList = cell(numMembers,1);
    
    for m = 1:numMembers
        dsTeM = toNetTableMSSAR(dsTest, dsTestSAR, ...
                                members(m).msBands, members(m).sarBand, inSz(1:2));
        [~, S] = classify(members(m).net, dsTeM, 'MiniBatchSize', miniBatchSize);
        scoresList{m} = S;
    end

    Savg = mean(cat(3, scoresList{:}), 3);
    [~, idx] = max(Savg, [], 2);

    % Get true labels from test set
    yTrue = gatherResponses(toNetTableMSSAR(dsTest, dsTestSAR, ...
                                            members(1).msBands, members(1).sarBand, inSz(1:2)));
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

function dsOut = toNetTableMSSAR(dsMS, dsSAR, msBands, sarBand, hw)
    % Create transform that handles struct array from combine
    fn = @(structArr) toRowMSSAR(structArr, msBands, sarBand, hw);
    dsOut = transform(combine(dsMS, dsSAR), fn);
end

function T = toRowMSSAR(structArray, msBands, sarBand, hw)
    % structArray is a 1x2 struct array: [sampleMS, sampleSAR]
    sampleMS = structArray(1);   % Access first struct
    sampleSAR = structArray(2);   % Access second struct
    
    % Extract 2 MS channels and 1 SAR channel
    XMS = sampleMS.X(:,:,msBands);      % 2 MS channels
    XSAR = sampleSAR.X(:,:,sarBand);    % 1 SAR channel

    % Combine into 3-channel image
    X3 = cat(3, XMS, XSAR);             % H x W x 3
    
    % Resize if needed
    if size(X3,1) ~= hw(1) || size(X3,2) ~= hw(2)
        X3 = imresize(X3, hw, 'bilinear');
    end
    
    % Return as table
    T = table({X3}, categorical(sampleMS.Label), 'VariableNames', {'input','response'});
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