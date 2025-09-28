addpath(genpath('matlab')); addpath('deployment');
load matlab/resRand.mat
load matlab/resRandRGB.mat
load matlab/resSAR.mat

opts.outDir = "deployment/onnx";
opts.modelNames = {'dense_rand','dense_randrgb','dense_sar'};
Export_ONNX({resRand,resRandRGB,resSAR}, opts);