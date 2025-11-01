function Isr = ESRGAN_2xSuperResolution(Ilr)
    % ESRGAN 2x Super-Resolution for 3-channel images
    persistent dlnG;
    
    if isempty(dlnG)
        fprintf('[INFO] Loading ESRGAN model once (persistent)...\n');
        % FIX: Make path absolute
        scriptDir = fileparts(mfilename('fullpath'));
        modelPath = fullfile(scriptDir, 'trained', 'ESRGAN100_RGB_Flickr2K_VGG54_2x_Generator_params_epoch300.mat');
        data = load(modelPath);
        dlnG = data.dlnG;
        
        if canUseGPU
            dlnG = dlupdate(@gpuArray, dlnG);
            fprintf('[INFO] ESRGAN model moved to GPU.\n');
        else
            fprintf('[WARN] GPU not available, running on CPU.\n');
        end
    end
    
    % Preprocess input
    Ilr_s = im2single(Ilr);
    Ilr_dl = dlarray(Ilr_s, 'SSCB');
    
    if canUseGPU
        Ilr_dl = gpuArray(Ilr_dl);
    end
    
    % Forward pass
    try
        Isr_dl = predict(dlnG, Ilr_dl);
    catch
        [Isr_dl, ~] = forward(dlnG, Ilr_dl);
    end
    
    % Postprocess
    Isr = gather(extractdata(Isr_dl));
    Isr = Isr * 0.5 + 0.5;     % rescale from [-1,1] → [0,1]
    Isr = max(0, min(1, Isr)); % clip
end



