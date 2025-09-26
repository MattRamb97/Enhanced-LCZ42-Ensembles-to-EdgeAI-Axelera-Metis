function train_teachers(MODE)
% MODE: 'RAND' | 'RANDRGB' | 'SAR' | 'ALL'
if nargin<1, MODE = 'ALL'; end
addpath(genpath('matlab')); rng(42,'twister');

dataRoot = 'data/lcz42';
load(fullfile(dataRoot,'tables_MS.mat'));   % train_MS, test_MS
load(fullfile(dataRoot,'tables_SAR.mat'));  % train_SAR, test_SAR

cfgCommon = struct;
cfgCommon.useZscore = true;
cfgCommon.useSARdespeckle = true;   % harmless for MS, active for SAR
cfgCommon.useAugmentation = true;
cfgCommon.reader = struct('type','custom', ...
                          'customFcn',@(row) h5_reader(row.Path,row.Index,row.Modality));

doRand    = any(strcmp(MODE, {'RAND','ALL'}));
doRandRGB = any(strcmp(MODE, {'RANDRGB','ALL'}));
doSAR     = any(strcmp(MODE, {'SAR','ALL'}));

if doRand
    fprintf('\n=== Training RAND (MS) ===\n');
    cfg = cfgCommon; cfg.trainTable = train_MS; cfg.testTable = test_MS;
    [dsTr, dsCal, dsTe, info] = DatasetReading(cfg);
    cfgT = struct('dsTrain',dsTr,'dsCal',dsCal,'dsTest',dsTe,'info',info, ...
                  'maxEpochs',25,'miniBatchSize',64,'learnRateMax',1e-3,'learnRateMin',1e-5);
    resRand = Rand_DenseNet(cfgT);
    save('matlab/resRand.mat','resRand','-v7.3');
end

if doRandRGB
    fprintf('\n=== Training RANDRGB (MS) ===\n');
    cfg = cfgCommon; cfg.trainTable = train_MS; cfg.testTable = test_MS;
    [dsTr, dsCal, dsTe, info] = DatasetReading(cfg);
    cfgT = struct('dsTrain',dsTr,'dsCal',dsCal,'dsTest',dsTe,'info',info, ...
                  'maxEpochs',25,'miniBatchSize',64,'learnRateMax',1e-3,'learnRateMin',1e-5);
    resRandRGB = RandRGB_DenseNet(cfgT);
    save('matlab/resRandRGB.mat','resRandRGB','-v7.3');
end

if doSAR
    fprintf('\n=== Training SAR (Sentinel-1) ===\n');
    cfg = cfgCommon; cfg.trainTable = train_SAR; cfg.testTable = test_SAR;
    [dsTr, dsCal, dsTe, info] = DatasetReading(cfg);
    cfgT = struct('dsTrain',dsTr,'dsCal',dsCal,'dsTest',dsTe,'info',info, ...
                  'maxEpochs',25,'miniBatchSize',64,'learnRateMax',1e-3,'learnRateMin',1e-5);
    resSAR = ensembleSARchannel_DenseNet(cfgT);
    save('matlab/resSAR.mat','resSAR','-v7.3');
end
end