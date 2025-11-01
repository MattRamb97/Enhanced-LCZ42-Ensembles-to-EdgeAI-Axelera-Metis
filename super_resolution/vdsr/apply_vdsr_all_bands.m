function X_sr = apply_vdsr_all_bands(X, vdsrNet, scaleFactor)
    % X: [32×32×10] in [0,255]
    % Output: [64×64×10] or [128×128×10]
    
    [H, W, C] = size(X);
    X_sr = zeros(H*scaleFactor, W*scaleFactor, C, 'single');
    
    for c = 1:C
        band = X(:,:,c) / 255;  % [0,1]
        
        % Bicubic upsample
        band_bicubic = imresize(band, scaleFactor, 'bicubic');
        
        % VDSR residual
        residual = predict(vdsrNet, band_bicubic);
        
        % Reconstruct
        band_sr = band_bicubic + double(residual);
        band_sr = min(1, max(0, band_sr));        % clip to [0,1]
        
        % Back to [0,255]
        X_sr(:,:,c) = band_sr * 255;
    end
    
    % Final clip and rescale back to original So2Sat LCZ42 format
    X_sr = max(0, min(255, X_sr));
    X_sr = (X_sr / 255) * 2.8;  % [0,255] → [0,2.8]
end