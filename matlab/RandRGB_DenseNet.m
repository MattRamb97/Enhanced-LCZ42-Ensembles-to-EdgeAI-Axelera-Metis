% RANDRGB_DENSENET  Train DenseNet201 on RandRGB 3-band images with upgrades.
%
% Usage:
%   cfg.trainTable = trainTable;   % table with Path,Label,CityID,Modality='MS'
%   cfg.testTable  = testTable;
%   [dsTrain, dsCal, dsTest, info] = DatasetReading(cfg);
%   cfg.dsTrain = dsTrain; cfg.dsCal = dsCal; cfg.dsTest = dsTest; cfg.info = info;
%   results = RandRGB_DenseNet(cfg);
%
% Matteo Rambaldi — Thesis utilities

function results = RandRGB_DenseNet(cfg)

arguments
    cfg struct
end

dsTrain = cfg.dsTrain;
dsCal   = cfg.dsCal;
dsTest  = cfg.dsTest;
info    = cfg.info;

numClasses   = info.numClasses;
classWeights = info.classWeights;
inputSize    = [32 32 3];

%% 1) Define DenseNet201 backbone
lgraph = layerGraph(densenet201);
newFCLayer = fullyConnectedLayer(numClasses,'Name','fc_final');
newSoftmax = softmaxLayer('Name','softmax');
newClass   = smoothCrossEntropy('sce',0.05,classWeights);

lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
lgraph = replaceLayer(lgraph,'fc1000_softmax',newSoftmax);
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClass);

net = dlnetwork(lgraph);

%% 2) Training parameters
maxEpochs   = cfgArg(cfg,'maxEpochs',25);
miniBatchSz = cfgArg(cfg,'miniBatchSize',64);
lrMax       = cfgArg(cfg,'learnRateMax',1e-3);
lrMin       = cfgArg(cfg,'learnRateMin',1e-5);
emaDecay    = 0.999;

mbq = minibatchqueue(dsTrain, ...
    'MiniBatchSize',miniBatchSz, ...
    'MiniBatchFcn',@(s) preprocessBatch(s,numClasses), ...
    'MiniBatchFormat',{'SSCB',''});

%% 3) Train loop with cosine LR + EMA
trLoss = []; valAcc = [];
emaNet = initEMA(net);

for epoch = 1:maxEpochs
    lr = cosineLR(epoch,maxEpochs,lrMax,lrMin);
    iteration = 0; reset(mbq)
    while hasdata(mbq)
        iteration = iteration + 1;
        [X,T] = next(mbq);

        [loss, grads] = dlfeval(@modelGradients, net, X, T);
        net = adamupdate(net, grads, iteration, lr);

        emaNet = updateEMA(emaNet, net, emaDecay);
        trLoss(end+1) = double(gather(extractdata(loss)));
    end
    valAcc(end+1) = evaluateAccuracy(net, dsCal, numClasses);
    fprintf('Epoch %d/%d | LR %.1e | TrainLoss %.4f | ValAcc %.3f\n', ...
        epoch, maxEpochs, lr, mean(trLoss(end-iteration+1:end)), valAcc(end));
end

% Swap in EMA weights
net = swapEMA(net, emaNet);

%% 4) Evaluation on TEST
[probs, yTrue] = forwardDatastore(net, dsTest, numClasses);

[~,yPred] = max(probs,[],2);
yPred = categorical(yPred,1:numClasses,info.classes);

figure; confusionchart(yTrue,yPred,'Normalization','row-normalized');

metrics = perClassROC(yTrue,probs,info.classes);

T = temperatureScale(logit(probs), yTrue);
probsCal = softmax(logit(probs)/T);
plotReliability(probsCal,yTrue);
plotRejection(probsCal,yTrue);

%% 5) Output struct
results.net   = net;
results.info  = info;
results.metrics = metrics;
results.temperature = T;

end

% ============================= HELPERS =============================

function [loss,grads] = modelGradients(net,X,T)
Y = forward(net,X);
loss = crossentropy(Y,T,'TargetCategories','independent');
grads = dlgradient(loss, net.Learnables);
end

function [X,T] = preprocessBatch(samples,numClasses)
X = cat(4,samples{:}.X);
X = single(X)/255;
T = onehotencode(categorical({samples{:}.Label}),1,'ClassNames',1:numClasses);
X = dlarray(gpuArray(X),'SSCB');
T = dlarray(gpuArray(T));
end

function lr = cosineLR(epoch,maxEpochs,lrMax,lrMin)
lr = lrMin + 0.5*(lrMax-lrMin)*(1+cos(pi*epoch/maxEpochs));
end

function acc = evaluateAccuracy(net, ds, numClasses)
[probs, yTrue] = forwardDatastore(net, ds, numClasses);
[~,yp] = max(probs,[],2);
acc = mean(double(yp)==grp2idx(yTrue));
end

function [probs, yTrue] = forwardDatastore(net, ds, numClasses)
reset(ds); probs = []; yTrue = [];
while hasdata(ds)
    s = read(ds);
    X = single(s.X)/255;
    X = dlarray(gpuArray(X),'SSCB');
    Y = predict(net,X);
    probs(end+1,:) = gather(extractdata(Y))'; %#ok<AGROW>
    yTrue(end+1,1) = s.Label; %#ok<AGROW>
end
end

function ema = initEMA(net)
p = net.Learnables;
ema = table(p.Layer,p.Parameter,p.Value,'VariableNames',["Layer","Parameter","Value"]);
end
function ema = updateEMA(ema, net, decay)
p = net.Learnables;
for i=1:height(p)
    ema.Value{i} = decay*ema.Value{i} + (1-decay)*p.Value{i};
end
end
function net = swapEMA(net, ema)
for i=1:height(ema)
    idx = strcmp(net.Learnables.Layer, ema.Layer{i}) & ...
          strcmp(net.Learnables.Parameter, ema.Parameter{i});
    net.Learnables.Value(idx) = ema.Value(i);
end
end

function m = perClassROC(yTrue,probs,classes)
m = struct();
for c = 1:numel(classes)
    [X,Y,~,AUC] = perfcurve(yTrue==classes(c),probs(:,c),true);
    m.(classes{c}).FPR = X; m.(classes{c}).TPR = Y; m.(classes{c}).AUC = AUC;
end
end

function Z = logit(P)
Z = log(max(P,1e-9)) - log(max(1-P,1e-9));
end

function P = softmax(Z)
expZ = exp(Z - max(Z,[],2));
P = expZ ./ sum(expZ,2);
end

function T = temperatureScale(logits,yTrue)
yIdx = grp2idx(yTrue);
nll = @(T) -mean(log( softmax(logits/T)(sub2ind(size(logits),1:numel(yIdx),yIdx')) + 1e-12 ));
T = fminbnd(nll,0.5,5);
end

function plotReliability(probs,yTrue)
M = 15;
[yProb,yPred] = max(probs,[],2);
acc = (yPred==grp2idx(yTrue));
edges = linspace(0,1,M+1);
[~,~,bin] = histcounts(yProb,edges);
binAcc = accumarray(bin,acc,[M 1],@mean,NaN);
binConf= accumarray(bin,yProb,[M 1],@mean,NaN);
figure; bar(binConf,binAcc-binConf);
ylabel('Gap (acc - conf)'); xlabel('Confidence'); title('Reliability Diagram');
end

function plotRejection(probs,yTrue)
[sortedMargin,idx] = sort(maxk(probs,2,2), 'descend');
pred = idx(:,1);
trueIdx = grp2idx(yTrue);
accs = []; covs = [];
for t = linspace(0,0.5,20)
    keep = sortedMargin(:,1)-sortedMargin(:,2) >= t;
    accs(end+1) = mean(pred(keep)==trueIdx(keep));
    covs(end+1) = mean(keep);
end
figure; plot(covs,accs,'-o'); xlabel('Coverage'); ylabel('Accuracy'); title('Rejection curve');
end