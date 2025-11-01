% ============================================================
%  TEST_FUSION_RESNET18_REAL.M
%  Runs an actual forward pass using LCZ42 data + TDA features
%
%  Requires:
%    - TDA/models/fusion_resnet18_*.onnx
%    - data/lcz42/training.h5
%    - TDA/tda_MS_features.npy and/or TDA/tda_SAR_features.npy
%
%  Matteo Rambaldi — Thesis utilities
% ============================================================

clear; clc;

% ----------- Select which model to test -----------
modelName = "fusion_resnet18_rand.onnx";   % or randrgb / ensemblesar
modelPath = fullfile("../TDA/models/",modelName);

if ~isfile(modelPath)
    error("Model not found: %s", modelPath);
end
fprintf("[INFO] Loading model: %s\n", modelPath);

% ----------- Import model -----------
net = importNetworkFromONNX(modelPath, ...
    "InputDataFormats", ["BCSS","BC"], ...
    "OutputDataFormats", "BC");

disp(net)

% ----------- Dataset paths -----------
dataRoot = fullfile("data","lcz42");
h5File   = fullfile(dataRoot, "training.h5");

isSAR = contains(modelName, "ensemblesar");
tdaPath = fullfile("TDA", iff(isSAR, "tda_SAR_features.npy", "tda_MS_features.npy"));

% ----------- Read one sample -----------
idx = 12345;   % any valid index within [1, 352366]
fprintf("[INFO] Reading sample #%d\n", idx);

% --- Read image patch ---
if isSAR
    % 2 MS + 1 SAR channel combination (like training)
    Xms  = h5_reader(h5File, idx, "MS");
    Xsar = h5_reader(h5File, idx, "SAR");
    Xms  = Xms ./ (2.8/255.0);
    Xsar = (max(min(Xsar,0.5),-0.5)+0.5)*255.0;
    msBands = [3 5]; sarBand = 4; % example
    img = cat(3, Xms(:,:,msBands), Xsar(:,:,sarBand));
else
    X = h5_reader(h5File, idx, "MS");
    X = X ./ (2.8/255.0);
    img = X(:,:,1:3);  % simplest RGB proxy
end

% --- Resize to ResNet input ---
img224 = imresize(img, [224 224], "bilinear");
% Convert (H×W×C) → (1×C×H×W) for ONNX "BCSS" layout
img224 = single(reshape(permute(img224, [3 1 2]), [1, size(img224,3), 224, 224]));
dlImg  = dlarray(img224, "BCSS");

% --- Load TDA vector ---
tdaMat = readNPY(tdaPath);        % requires npy-matlab on path
tdaVec = single(tdaMat(idx,:));
dlTDA = dlarray(tdaVec(:)', "BC");   % [1×18000], batch=1, channel=18000

% ----------- Forward pass -----------
fprintf("[INFO] Running forward pass...\n");
dlOut = predict(net, dlImg, dlTDA);

out = extractdata(dlOut);
[~, predIdx] = max(out);
fprintf("[RESULT] Predicted LCZ class: %d\n", predIdx);
fprintf("[RESULT] Sum=%.4f | Max=%.4f | Min=%.4f\n", ...
    sum(out(:)), max(out(:)), min(out(:)));

% Apply softmax
probs = softmax(dlOut);
[conf, predIdx] = max(probs);
fprintf("[RESULT] Predicted LCZ class: %d (confidence = %.3f)\n", predIdx, conf);

disp(" Real forward pass successful!");

% ----------- Local helper (MATLAB <R2025b has no iff) -----------
function r = iff(cond, a, b)
if cond, r = a; else, r = b; end
end