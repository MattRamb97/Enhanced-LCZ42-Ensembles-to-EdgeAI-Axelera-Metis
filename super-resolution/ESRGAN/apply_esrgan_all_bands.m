function X_sr = apply_esrgan_all_bands(X, scaleFactor)
    % X: [32×32×10] in [0,255]
    % Output: [64×64×10] (for 2x)
    
    [H, W, ~] = size(X);
    X_sr = zeros(H*scaleFactor, W*scaleFactor, 10, 'single');
    bandWritten = false(1, 10);
    
    % Semantic triplets
    triplets = {
        [3,2,1],    % RGB: B4,B3,B2
        [7,5,4],    % Vegetation: B8(NIR), B6(RedEdge), B5(RedEdge)
        [9,10,8],   % Moisture: B11(SWIR1), B12(SWIR2), B8A(NIR-narrow)
        [10,10,6]   % Filler: B12, B12, B7
    };
    
    for t = 1:numel(triplets)
        bands = triplets{t};
        X_triplet = X(:,:,bands);
        
        % per-channel normalization to [0,1]
        for ch = 1:3 
            ch_min = min(X_triplet(:,:,ch), [], 'all');
            ch_max = max(X_triplet(:,:,ch), [], 'all');
            if ch_max > ch_min
                X_triplet(:,:,ch) = (X_triplet(:,:,ch) - ch_min) / (ch_max - ch_min);  % → [0,1]
            else
                X_triplet(:,:,ch) = 0;  % Flat band
            end
        end
        % Convert to [0,255]
        X_triplet = uint8(X_triplet * 255);

        X_sr_triplet = ESRGAN_2xSuperResolution(X_triplet);
    
        for i = 1:3
            bandIdx = bands(i);
            if bandIdx <= 10 && ~bandWritten(bandIdx)
                X_sr(:,:,bandIdx) = X_sr_triplet(:,:,i); %[0,1] -> [0,255]
                bandWritten(bandIdx) = true;
            end
        end
    end
end