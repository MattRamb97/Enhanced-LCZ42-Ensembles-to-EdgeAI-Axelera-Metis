% test_rand_local.m - Full test with real H5 files on Mac

clear all; clc;

%% Setup - Point to your H5 files
H5_DIR = '/Users/matteo.rambaldi/Library/CloudStorage/OneDrive-AXELERAAI/Documents/GitHub/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/data/lcz42/';  % CHANGE THIS to where your files are
cd(H5_DIR);

%% 1. Create tables from real H5 (small subset)
fprintf('=== Creating tables ===\n');
% Your make_tables_from_h5 function, but we'll just create manually for testing

tr = fullfile(H5_DIR, 'training.h5');
te = fullfile(H5_DIR, 'testing.h5');

% Read just the labels to get structure
labels_tr = h5read(tr, '/label', [1 1], [17 500]);  % First 500 samples
labels_te = h5read(te, '/label', [1 1], [17 100]);  % First 100 samples

% Convert to categorical
[~, idx_tr] = max(labels_tr, [], 1);
[~, idx_te] = max(labels_te, [], 1);

train_MS = table(repmat(string(tr), 500, 1), ...
                 categorical(idx_tr'), ...
                 (1:500)', ...
                 repmat("MS", 500, 1), ...
                 'VariableNames', {'Path','Label','Index','Modality'});

test_MS = table(repmat(string(te), 100, 1), ...
                categorical(idx_te'), ...
                (1:100)', ...
                repmat("MS", 100, 1), ...
                'VariableNames', {'Path','Label','Index','Modality'});

fprintf('train_MS: %d samples, %d classes\n', height(train_MS), numel(categories(train_MS.Label)));
fprintf('test_MS: %d samples\n', height(test_MS));

%% 2. Create datastores (minimal DatasetReading simulation)
fprintf('\n=== Creating datastores ===\n');

cfg = struct();
cfg.trainTable = train_MS;
cfg.testTable = test_MS;
cfg.useZscore = false;
cfg.useAugmentation = false;
cfg.reader = struct('type', 'custom', 'customFcn', @(row) h5_reader(row.Path, row.Index, row.Modality));

% You need h5_reader.m in your path
addpath(genpath('matlab'));  % Adjust path

[dsTr, dsTe, info] = DatasetReading(cfg);

fprintf('info.numClasses: %d\n', info.numClasses);
fprintf('info.classes type: %s\n', class(info.classes));

%% 3. Call Rand_DenseNet with tiny config
fprintf('\n=== Training tiny ensemble ===\n');

cfgT = struct();
cfgT.dsTrain = dsTr;
cfgT.dsTest = dsTe;
cfgT.info = info;
cfgT.numMembers = 2;      % Just 2 members
cfgT.maxEpochs = 2;       % Just 2 epochs
cfgT.miniBatchSize = 16;  % Small batch
cfgT.learnRate = 1e-3;

try
    res = Rand_DenseNet(cfgT);
    fprintf('\n✅ COMPLETE SUCCESS!\n');
    fprintf('Test Top-1: %.4f\n', res.testTop1);
    fprintf('Members: %d\n', numel(res.members));
    fprintf('\nNetwork construction and training pipeline verified.\n');
    fprintf('Ready to deploy to cluster.\n');
catch ME
    fprintf('\n❌ FAILED\n');
    fprintf('Error: %s\n', ME.message);
    disp(getReport(ME, 'extended'));
end