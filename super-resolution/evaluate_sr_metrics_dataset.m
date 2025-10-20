function metrics = evaluate_sr_metrics_dataset(h5_orig, h5_sr, modality)
    % Evaluate PSNR, SSIM, RMSE over an entire dataset
    % modality: "MS" or "SAR"
    
    % Map modality to dataset name
    switch upper(modality)
        case "MS"
            datasetName = '/sen2';
            numBands = 10;
        case "SAR"
            datasetName = '/sen1';
            numBands = 8;
        otherwise
            error("Unsupported modality: %s", modality);
    end
    
    info = h5info(h5_sr, datasetName); % h5_orig
    N = info.Dataspace.Size(4);  % Number of patches
    fprintf('Evaluating %d samples...\n', N, modality);
    
    % Scaling [0,2.8] → [0,255]
    scale = 255 / 2.8;

    psnr_vals = zeros(N, numBands);
    ssim_vals = zeros(N, numBands);
    rmse_vals = zeros(N, numBands);

    for i = 1:N
        if mod(i,1000)==0, fprintf('  Sample %d / %d\n', i, N); end
        
        X_hr = h5_reader(h5_orig, i, modality); % 32×32xc  
        X_sr = h5_reader(h5_sr, i, modality);   % 64×64xc     
        
        X_hr = imresize(X_hr, [size(X_sr,1), size(X_sr,2)], 'bicubic') * scale;
        X_sr = X_sr * scale;
        
        for c = 1:numBands
            [A, B] = normalize_pair(im2double(X_sr(:,:,c)), im2double(X_hr(:,:,c)));

            psnr_vals(i,c) = psnr(A, B);
            ssim_vals(i,c) = ssim(A, B);
            rmse_vals(i,c) = sqrt(mean((A(:) - B(:)).^2));
        end
    end

    % Save metrics
    metrics.PSNR  = psnr_vals;
    metrics.SSIM  = ssim_vals;
    metrics.RMSE  = rmse_vals;

    % Compute mean over all samples and bands
    metrics.MeanPSNR = mean(psnr_vals(:));
    metrics.MeanSSIM = mean(ssim_vals(:));
    metrics.MeanRMSE = mean(rmse_vals(:));
end

% Helper

function [A_norm, B_norm] = normalize_pair(A, B)
    low = min(min(B), min(A));
    high = max(max(B), max(A));
    if high > low
        A_norm = (A - low) / (high - low);
        B_norm = (B - low) / (high - low);
    else
        A_norm = zeros(size(A));
        B_norm = zeros(size(B));
    end
end