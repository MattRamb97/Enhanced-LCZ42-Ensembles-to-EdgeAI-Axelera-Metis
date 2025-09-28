% FUSION_ENSEMBLE  Combine Rand, RandRGB, and SAR DenseNet models.
%
% Usage:
%   cfg.models = {resRand, resRandRGB, resSAR}; % structs from training
%   cfg.dsCal  = dsCal;   % calibration datastore
%   cfg.dsTest = dsTest;  % test datastore
%   fusionResults = Fusion_Ensemble(cfg);

function fusionResults = Fusion_Ensemble(cfg)

arguments
    cfg struct
end

models = cfg.models;
dsCal  = cfg.dsCal;
dsTest = cfg.dsTest;
info   = models{1}.info;
numClasses = info.numClasses;

%% 1) Collect logits/probs on CAL and TEST for all models
[logitsCal, yCal] = collectLogits(models, dsCal, numClasses);
[logitsTest, yTest] = collectLogits(models, dsTest, numClasses);

%% 2) Fusion strategies
% --- (a) Simple sum-rule (average probs)
probsSum = mean(softmax(logitsTest),3);

% --- (b) Temperature scaling (per model, then average)
T = zeros(1,numel(models));
for m = 1:numel(models)
    T(m) = temperatureScale(logitsCal(:,:,m), yCal);
end
probsTemp = zeros(size(probsSum));
for m = 1:numel(models)
    probsTemp = probsTemp + softmax(logitsTest(:,:,m)/T(m));
end
probsTemp = probsTemp / numel(models);

% --- (c) Logistic regression stacking (learned fusion)
Xcal = reshape(softmax(logitsCal), size(logitsCal,1), []);
[B,~,~] = mnrfit(Xcal, categorical(yCal));
Xtest = reshape(softmax(logitsTest), size(logitsTest,1), []);
probsStack = mnrval(B,Xtest);

%% 3) Evaluation
fusionResults.sum     = evaluateFusion(probsSum,yTest,info);
fusionResults.temp    = evaluateFusion(probsTemp,yTest,info);
fusionResults.stacking= evaluateFusion(probsStack,yTest,info);

fprintf('Fusion Accuracies: Sum=%.3f | Temp=%.3f | Stacking=%.3f\n', ...
    fusionResults.sum.acc, fusionResults.temp.acc, fusionResults.stacking.acc);

end

% ====================== HELPERS ======================

function [logits,yTrue] = collectLogits(models, ds, numClasses)
reset(ds); logits = []; yTrue = [];
while hasdata(ds)
    s = read(ds);
    X = single(s.X)/255;
    X = dlarray(gpuArray(X),'SSCB');
    batchLogits = [];
    for m = 1:numel(models)
        net = models{m}.net;
        Y = predict(net,X);
        batchLogits(:,:,m) = gather(extractdata(Y)); %#ok<AGROW>
    end
    logits(end+1,:,:) = batchLogits; %#ok<AGROW>
    yTrue(end+1,1) = s.Label; %#ok<AGROW>
end
end

function P = softmax(Z)
expZ = exp(Z - max(Z,[],2));
P = expZ ./ sum(expZ,2);
end

function res = evaluateFusion(probs,yTrue,info)
[~,yPred] = max(probs,[],2);
yPred = categorical(yPred,1:info.numClasses,info.classes);
acc = mean(yPred==yTrue);

figure; confusionchart(yTrue,yPred,'Normalization','row-normalized');
title(sprintf('Confusion Matrix (Acc=%.3f)',acc));

% ROC AUC per class
metrics = struct();
for c = 1:numel(info.classes)
    [X,Y,~,AUC] = perfcurve(yTrue==info.classes(c),probs(:,c),true);
    metrics.(info.classes{c}).AUC = AUC;
end

res.acc = acc;
res.metrics = metrics;
end

function T = temperatureScale(logits,yTrue)
yIdx = grp2idx(yTrue);
nll = @(T) -mean(log( softmax(logits/T)(sub2ind(size(logits),1:numel(yIdx),yIdx')) + 1e-12 ));
T = fminbnd(nll,0.5,5);
end