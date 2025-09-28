addpath(genpath('matlab'));
load matlab/resRand.mat
load matlab/resRandRGB.mat
load matlab/resSAR.mat

% Recreate MS datastores (for calibration/test)
dataRoot = 'data/lcz42';
load(fullfile(dataRoot,'tables_MS.mat'));   % train_MS, test_MS
cfg = struct('trainTable',train_MS,'testTable',test_MS, ...
             'useZscore',true,'useSARdespeckle',true,'useAugmentation',true);
cfg.reader = struct('type','custom','customFcn',@(row) h5_reader(row.Path,row.Index,row.Modality));
[~, dsCal_MS, dsTest_MS] = DatasetReading(cfg); %#ok<ASGLU>

cfgF.models = {resRand,resRandRGB,resSAR};
cfgF.dsCal  = dsCal_MS;
cfgF.dsTest = dsTest_MS;
fusionResults = Fusion_Ensemble(cfgF);
save('matlab/fusionResults.mat','fusionResults','-v7.3');