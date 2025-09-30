addpath('matlab');  % your repo path
S = load('data/lcz42/tables_SAR.mat');
cfg = struct;
cfg.trainTable = S.train_SAR; cfg.testTable = S.test_SAR;
cfg.reader.type='custom';
cfg.reader.customFcn=@(row) h5_reader(row.Path,row.Index,row.Modality);
[dsTr, dsCal, dsTe, info] = DatasetReading(cfg);

cfgT = struct('dsTrain',dsTr,'dsCal',dsCal,'dsTest',dsTe,'info',info, ...
              'numMembers',10,'maxEpochs',7,'miniBatchSize',50,'plots',"none");

resSAR = ensembleSARchannel_DenseNet(cfgT);
save('results_sar.mat','resSAR','-v7.3');