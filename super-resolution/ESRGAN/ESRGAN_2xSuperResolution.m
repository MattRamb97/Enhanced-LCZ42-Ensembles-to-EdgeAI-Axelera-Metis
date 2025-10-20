function Isr = ESRGAN_2xSuperResolution(Ilr)
    persistent dlnG;
    if isempty(dlnG)
        fprintf('[INFO] Loading ESRGAN model once (persistent)...\n');
        data = load('trained/ESRGAN100_RGB_Flickr2K_VGG54_2x_Generator_params_epoch300.mat');
        dlnG = data.dlnG;

        % Try moving to GPU once if available
        if canUseGPU
            dlnG = dlupdate(@gpuArray, dlnG);
            fprintf('[INFO] ESRGAN model moved to GPU.\n');
        else
            fprintf('[WARN] GPU not available, running on CPU.\n');
        end
    end

    Ilr_s = im2single(Ilr);
    Ilr_dl = dlarray(Ilr_s, 'SSCB');   % (H,W,C,B)

    % Run inference on GPU
    if canUseGPU
        Ilr_dl = gpuArray(Ilr_dl);
    end

    % --- Forward pass
    try
        Isr_dl = predict(dlnG, Ilr_dl);   % predict preferred over forward
    catch
        % For older models that need 'forward'
        [Isr_dl, ~] = forward(dlnG, Ilr_dl);
    end

    Isr = gather(extractdata(Isr_dl));
    Isr = Isr * 0.5 + 0.5;  
    Isr = max(0, min(1, Isr));
end

// function Isr = ESRGAN_2xSuperResolution(Ilr)
//     scale = 2;

//     % SRGAN trained network ==> dlnG
//     load('trained/ESRGAN100_RGB_Flickr2K_VGG54_2x_Generator_params_epoch300.mat');

//     % Ilrを超解像しIsrを作る。
    
//     Ilr_s = im2single(Ilr);
//     Ilr_dl = dlarray(Ilr_s, 'SSCB');
    
//     [Isr_dl, stateG] = forward(dlnG, Ilr_dl);
    
//     Isr = single(extractdata(Isr_dl));
//     Isr = Isr * 0.5 + 0.5;    

//     Isr = max(0, min(1, Isr));
//     %figure;
//     %imshow(Isr);
//     %title('ESRGAN Super Resolution Image');
// end





