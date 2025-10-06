function train_teachers(MODE)
    % MODE: 'RAND' | 'RANDRGB' | 'SAR' | 'ALL'
    if nargin<1, MODE = 'ALL'; end
    addpath(genpath('matlab')); rng(42,'twister');
    EnableGPU(1);
    
    dataRoot = 'data/lcz42';
    load(fullfile(dataRoot,'tables_MS.mat'), 'train_MS', 'test_MS');
    load(fullfile(dataRoot,'tables_SAR.mat'), 'train_SAR', 'test_SAR');
    
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
        [dsTr, dsTe, info] = DatasetReading(cfg);
        cfgT = struct('dsTrain',dsTr,'dsTest',dsTe,'info',info, ...
                      'maxEpochs',12,'miniBatchSize',128,'learnRate',1e-3);
        resRand = Rand_DenseNet(cfgT);
        resRand.name = "Rand";                                  
        save('matlab/resRand.mat','resRand','-v7.3');
    end
    
    if doRandRGB
        fprintf('\n=== Training RANDRGB (MS) ===\n');
        cfg = cfgCommon; cfg.trainTable = train_MS; cfg.testTable = test_MS;
        [dsTr, dsTe, info] = DatasetReading(cfg);
        cfgT = struct('dsTrain',dsTr,'dsTest',dsTe,'info',info, ...
                      'maxEpochs',12,'miniBatchSize',128,'learnRate',1e-3);
        resRandRGB = RandRGB_DenseNet(cfgT);
        resRandRGB.name = "RandRGB";                            
        save('matlab/resRandRGB.mat','resRandRGB','-v7.3');
    end
    
    if doSAR
        fprintf('\n=== Training SAR (Sentinel-1) ===\n');
        cfg = cfgCommon; cfg.trainTable = train_SAR; cfg.testTable = test_SAR;
        [dsTr, dsTe, info] = DatasetReading(cfg);
        cfgT = struct('dsTrain',dsTr,'dsTest',dsTe,'info',info, ...
                      'maxEpochs',12,'miniBatchSize',128,'learnRate',1e-3);
        resSAR = ensembleSARchannel_DenseNet(cfgT);
        resSAR.name = "SAR";                                     
        save('matlab/resSAR.mat','resSAR','-v7.3');
    end
end