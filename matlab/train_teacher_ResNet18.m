% TRAIN_TEACHERS_V2  Main training orchestrator with MS+SAR fusion support.
%
% Trains all three ensembles:
%   - RAND     : Random 3-band MS
%   - RANDRGB  : Mixed MS ensemble (2 MS random + 1 Fixed RGB MS band)
%   - SAR      : Mixed MS+SAR ensemble (2 MS + 1 SAR bands per member)
%
% Usage:
%   train_teachers_v2('RAND')     - Train MS random band ensemble
%   train_teachers_v2('RANDRGB')  - Train MS RGB-only ensemble
%   train_teachers_v2('SAR')      - Train MS+SAR ensemble (2 MS + 1 SAR)
%   train_teachers_v2('ALL')      - Train all three (default)
%
% Assumes HDF5 tables are already saved in:
%   data/lcz42/tables_MS.mat → train_MS, test_MS
%   data/lcz42/tables_SAR.mat → train_SAR, test_SAR
%
% Ensembles are trained with 12 epochs, batch size 128 by default.
% Z-score normalization and augmentation are enabled.
%
% Outputs (in 'matlab/' folder):
%   - resRand.mat
%   - resRandRGB.mat
%   - resSAR.mat
%
% Author: Matteo Rambaldi — Thesis utilities

function train_teacher_ResNet18(MODE)
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
        resRand = Rand_ResNet18(cfgT);
        resRand.name = "Rand";                                  
        save('matlab/resRand_resnet18.mat', 'resRand', '-v7.3');
    end
    
    if doRandRGB
        fprintf('\n=== Training RANDRGB (MS) ===\n');
        cfg = cfgCommon; cfg.trainTable = train_MS; cfg.testTable = test_MS;
        [dsTr, dsTe, info] = DatasetReading(cfg);
        cfgT = struct('dsTrain',dsTr,'dsTest',dsTe,'info',info, ...
                      'maxEpochs',12,'miniBatchSize',128,'learnRate',1e-3);
        resRandRGB = RandRGB_ResNet18(cfgT);
        resRandRGB.name = "RandRGB";                            
        save('matlab/resRandRGB_resnet18.mat', 'resRandRGB', '-v7.3');
    end
    
    if doSAR
        % Need BOTH MS and SAR datastores
        cfgMS = cfgCommon; 
        cfgMS.trainTable = train_MS; 
        cfgMS.testTable = test_MS;
        [dsTrMS, dsTeMS, infoMS] = DatasetReading(cfgMS);
        
        cfgSAR = cfgCommon; 
        cfgSAR.trainTable = train_SAR; 
        cfgSAR.testTable = test_SAR;
        [dsTrSAR, dsTeSAR, infoSAR] = DatasetReading(cfgSAR);
        % Pass both to the ensemble function
        cfgT = struct('dsTrain',dsTrMS, 'dsTest',dsTeMS, ...
                      'dsTrainSAR',dsTrSAR, 'dsTestSAR',dsTeSAR, ...
                      'info',infoMS, ...
                      'maxEpochs',12,'miniBatchSize',128,'learnRate',1e-3);
        resSAR = resSAR_resnet18(cfgT);
        resSAR.name = "SAR";                                     
        save('matlab/resSAR_resnet18.mat', 'resSAR', '-v7.3');
    end
end
