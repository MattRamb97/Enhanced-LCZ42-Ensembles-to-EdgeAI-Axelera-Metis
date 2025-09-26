% EXPORT_ONNX  Export trained MATLAB dlnetworks to ONNX + metadata.
%
% Usage:
%   models = {resRand, resRandRGB, resSAR};  % structs from your training scripts
%   opts.outDir = 'deployment/onnx';
%   opts.modelNames = {'dense_rand','dense_randrgb','dense_sar'}; % optional
%   out = Export_ONNX(models, opts);
%
% Notes:
% - Exports ONNX with input format = BCSS (batch, channel, height, width) → ONNX NCHW
% - Saves preprocessing metadata (paper scaling + optional z-score mu/sigma)
% - Saves class names mapping
% - Does NOT bake temperature scaling or fusion — do that at runtime in Python.
%
% Matteo Rambaldi — Thesis utilities

function out = Export_ONNX(models, opts)

arguments
    models (1,:) cell
    opts.outDir (1,1) string = "deployment/onnx"
    opts.modelNames (1,:) cell = {}
    opts.opset (1,1) double = 17  % safe default for many runtimes
end

if ~exist(opts.outDir, 'dir'); mkdir(opts.outDir); end

% derive names
if isempty(opts.modelNames)
    opts.modelNames = arrayfun(@(k) sprintf("model_%02d",k), 1:numel(models), 'uni',0);
end

% export each model
filenames = strings(1,numel(models));
for k = 1:numel(models)
    M = models{k};
    assert(isfield(M,'net') && isa(M.net,'dlnetwork'), 'models{%d}.net must be a dlnetwork', k);

    % ONNX filename
    onnxName = opts.modelNames{k} + ".onnx";
    onnxPath = fullfile(opts.outDir, onnxName);
    
    % Export:
    %  - InputDataFormats 'BCSS': (Batch, Channel, Spatial, Spatial) → NCHW
    %  - OutputDataFormats 'BC' : class scores per sample
    fprintf('[Export_ONNX] Exporting %s ...\n', onnxPath);
    exportONNXNetwork(M.net, onnxPath, ...
        'OpsetVersion', opts.opset, ...
        'InputDataFormats','BCSS', ...
        'OutputDataFormats','BC');

    filenames(k) = string(onnxPath);
end

% ---- Save preprocessing metadata (one JSON for all models) ----
% Pull info from the first model (they share protocol)
info = models{1}.info;

meta = struct();
meta.input = struct('layout','NCHW', 'dtype','float32', 'shape',[1,3,32,32]); % batch=dynamic in ONNX
meta.scaling = struct();
meta.scaling.optical = struct('type','linear','src_range',[0,2.8],'dst_range',[0,255]); % paper scaling
meta.scaling.sar     = struct('type','clip_linear','clip',[-0.5,0.5],'dst_range',[0,255]); 
meta.zscore = struct('enabled', ~isempty(info.mu) && ~isempty(info.sigma), ...
                     'mu', double(info.mu(:)'), 'sigma', double(info.sigma(:)'));
meta.labels = cellstr(info.classes(:));
meta.note   = ['Inputs expected as 3x32x32 float32. If you used paper scaling to [0,255], ' ...
               'normalize to [0,1] before feeding the ONNX (i.e., divide by 255). ' ...
               'If z-score enabled, apply (x - mu)/sigma per-channel after scaling.'];

% write JSON
metaPath = fullfile(opts.outDir, "preprocessing.json");
fid = fopen(metaPath,'w'); fwrite(fid, jsonencode(meta, 'PrettyPrint',true)); fclose(fid);

% also write labels.txt (one class per line)
labelsPath = fullfile(opts.outDir, "labels.txt");
fid = fopen(labelsPath,'w');
for i=1:numel(info.classes), fprintf(fid, '%s\n', info.classes{i}); end
fclose(fid);

out = struct();
out.files = filenames;
out.preprocessing = string(metaPath);
out.labels = string(labelsPath);

fprintf('[Export_ONNX] Done.\n');
end