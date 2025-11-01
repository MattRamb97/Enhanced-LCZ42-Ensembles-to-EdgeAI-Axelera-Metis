% ============================================================
%  TEST_FUSION_RESNET18_ENSEMBLE_TEST.M
%  Evaluate a fusion ResNet18+TDA ONNX model on the 10 *testing* datasets
%
%  Requires:
%    - models/exported/fusion_resnet18_{rand|randrgb|ensemblesar}.onnx
%    - ../data/lcz42/testing*.h5   (baseline + SR variants)
%    - data/tda_{MS|SAR}_features_test.npy
%    - data/labels_test.npy
% ============================================================

clear; clc;

fusionType = "rand";   % <- "rand" or "randrgb" | "ensemblesar"

% You are in TDA/scripts/
projectRoot = pwd;  % TDA/scripts/
tdaRoot = fileparts(projectRoot);  % Go to TDA/
modelsDir = fullfile(tdaRoot, "models", "exported");
tdaDataDir = fullfile(tdaRoot, "data");
dataRoot = fullfile(fileparts(tdaRoot), "data", "lcz42");  % Go up from TDA to main project, then data/lcz42
resultsDir = fullfile(tdaRoot, "results");

if ~isfolder(resultsDir), mkdir(resultsDir); end

switch fusionType
    case "rand"
        onnxName = "fusion_resnet18_rand.onnx";
        tdaPath = fullfile(tdaDataDir, "tda_MS_features_test.h5");  % H5 file
        tdaDataset = '/tda_MS_features_test';  % Dataset name inside H5
        memberKind = "MS";
    case "randrgb"
        onnxName = "fusion_resnet18_randrgb.onnx";
        tdaPath = fullfile(tdaDataDir, "tda_MS_features_test.h5");  
        tdaDataset = '/tda_MS_features_test';
        memberKind = "MS";
    case "ensemblesar"
        onnxName = "fusion_resnet18_ensemblesar.onnx";
        tdaPath = fullfile(tdaDataDir, "tda_SAR_features_test.h5");
        tdaDataset = '/tda_SAR_features_test';
        memberKind = "MSSAR";
end

modelPath = fullfile(modelsDir, onnxName);
assert(isfile(modelPath), "ONNX not found: %s", modelPath);
assert(isfile(tdaPath), "TDA TEST features not found: %s", tdaPath);

fprintf("[INFO] Loading model: %s\n", modelPath);
net = importNetworkFromONNX(modelPath, ...
    "InputDataFormats", ["BCSS","BC"], ...
    "OutputDataFormats","BC");

%% -------- Load TDA + labels for TEST split --------
% Read from H5 files
tda = single(h5read(tdaPath, tdaDataset));  % Read from H5
labels = int32(h5read(fullfile(tdaDataDir, "labels_test.h5"), '/labels'));  % Read from H5

Ntest = size(tda,1);
assert(length(labels)==Ntest, "Label count does not match TDA samples.");

evalN = Ntest;
rng(42,'twister');
idxList = randperm(Ntest, evalN);

%% -------- TEST datasets --------
%{
    Dataset file               Member name
    -------------------------  ----------------
    testing.h5                 Baseline_1
    testing.h5                 Baseline_2
    testing_vdsr2x.h5          VDSR2x
    testing_edsr2x.h5          EDSR2x
    testing_esrgan2x.h5        ESRGAN2x
    testing_edsr4x.h5          EDSR4x
    testing_swinir2x.h5        SwinIR2x
    testing_vdsr3x.h5          VDSR3x
    testing_bsrnet2x.h5        BSRNet2x
    testing_realesrgan4x.h5    RealESRGAN4x
%}
datasets = {
    "testing.h5"                 "Baseline_1"
    "testing.h5"                 "Baseline_2"
    "testing_vdsr2x.h5"          "VDSR2x"
    "testing_edsr2x.h5"          "EDSR2x"
    "testing_esrgan2x.h5"        "ESRGAN2x"
    "testing_edsr4x.h5"          "EDSR4x"
    "testing_swinir2x.h5"        "SwinIR2x"
    "testing_vdsr3x.h5"          "VDSR3x"
    "testing_bsrnet2x.h5"        "BSRNet2x"
    "testing_realesrgan4x.h5"    "RealESRGAN4x"
    % (add your 8 SR datasets below)
};

classes = 1:17;
summary = table('Size',[size(datasets,1) 3], ...
    'VariableTypes',["string","double","double"], ...
    'VariableNames',["Dataset","Top1","N"]);

results = struct([]);

baseH5 = fullfile(dataRoot, "testing.h5"); % always used for SAR fallback

%% -------- Evaluation loop --------
for i = 1:size(datasets,1)
    h5File = fullfile(dataRoot, datasets{i,1});
    name   = datasets{i,2};
    assert(isfile(h5File), "Missing H5: %s", h5File);

    fprintf("\n[MEMBER %02d] %s\n", i, name);
    correct = 0;
    yPred = zeros(evalN,1,'int32');
    yTrue = zeros(evalN,1,'int32');

    for k = 1:evalN
        idx = idxList(k);

        % --- Build image exactly as in training ---
        if memberKind == "MS"
            Xms = h5_reader(h5File, idx, "MS");  % (32,32,10)
            switch fusionType
                case "rand"
                    % Training: raw MS → /255.0
                    bands = randperm(10, 3);
                    img = single(Xms(:,:,bands)) / 255.0;
        
                case "randrgb"
                    % Training: (MS / (2.8/255)) → clamp [0,255] → /255.0
                    patch = (Xms / (2.8 / 255.0));
                    patch = min(max(patch, 0), 255);
                    rgb = randsample([3 2 1], 1);  % B4,B3,B2
                    others = randsample(setdiff(1:10, rgb), 2, false);
                    img = single(patch(:,:, [others rgb])) / 255.0;
            end
        
        else % memberKind == "MSSAR"
            % MS from current test file (SR or baseline)
            Xms = h5_reader(h5File, idx, "MS");

            % SAR from current file if present, else from base testing.h5
            if hasH5Dataset(h5File, 'sen1')
                Xsar = h5_reader(h5File, idx, "SAR");
            else
                Xsar = h5_reader(baseH5,  idx, "SAR");
            end

            % --- Ensure SAR and MS have matching spatial sizes ---
            [hMS, wMS, ~] = size(Xms);
            [hSAR, wSAR, ~] = size(Xsar);
            if hMS ~= hSAR || wMS ~= wSAR
                Xsar = imresize(Xsar, [hMS, wMS], "bilinear");
            end

            % Match training preprocessing
            Xms  = (Xms / (2.8 / 255.0));
            Xms  = min(max(Xms, 0), 255);
            Xsar = (max(min(Xsar, 0.5), -0.5) + 0.5) * 255.0;
            Xsar = single(Xsar);

            ms_selected  = randperm(10, 2);
            sar_selected = randi(8);
            img = cat(3, Xms(:,:,ms_selected), Xsar(:,:,sar_selected));
            img = single(img) / 255.0;
        end

        % --- Format for ONNX ---
        img224 = imresize(img,[224 224],"bilinear");
        img224 = single(reshape(permute(img224,[3 1 2]),[1,size(img224,3),224,224]));
        dlImg  = dlarray(img224,"BCSS");
        dlTDA  = dlarray(tda(idx,:), "BC");

        % --- Forward ---
        dlOut = predict(net, dlImg, dlTDA);
        probs = softmax(dlOut);
        [~, pred] = max(probs);

        % --- Ground truth from labels_test.npy ---
        yTrue(k) = labels(idx);
        yPred(k) = int32(extractdata(pred));

        correct = correct + (yPred(k)==yTrue(k));
    end

    top1 = correct / evalN;
    fprintf("  Top-1 over %d samples = %.4f\n", evalN, top1);

    results(i).Dataset = name;
    results(i).Top1    = top1;
    results(i).N       = evalN;
    results(i).yPred   = yPred;
    results(i).yTrue   = yTrue;

    summary.Dataset(i) = name;
    summary.Top1(i)    = top1;
    summary.N(i)       = evalN;
end

%% -------- Save all --------
stem = sprintf("fusion_%s_eval_TEST", fusionType);
save(fullfile(resultsDir, stem + ".mat"), "summary","results");
writetable(summary, fullfile(resultsDir, stem + ".csv"));

fprintf("\n[✓] Saved: %s and %s\n", ...
    fullfile(resultsDir, stem + ".mat"), ...
    fullfile(resultsDir, stem + ".csv"));

%% -------- Helper Function --------

function tf = hasH5Dataset(h5file, dset)
    try
        h5info(h5file, ['/' dset]);  % throws if missing
        tf = true;
    catch
        tf = false;
    end
end