% deployment/SelfTest_ONNX.m
function SelfTest_ONNX(onnxPath)
fprintf('Round-trip check for %s\n', onnxPath);
net = importONNXNetwork(onnxPath, 'OutputLayerType','classification', 'ImportWeights',true);
disp(net.Layers(1));
end