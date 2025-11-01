function X_sr = apply_esrgan_all_bands(X, scaleFactor)
    % X: [32×32×10] in [0,255]
    % Output: [64×64×10] in [0,255] - PRESERVING ORIGINAL RANGES
    
    [H, W, ~] = size(X);
    X_sr = zeros(H*scaleFactor, W*scaleFactor, 10, 'single');
    bandWritten = false(1, 10);
    
    triplets = {
        [3,2,1],    % RGB: B4,B3,B2
        [7,5,4],    % Vegetation: B8(NIR), B6(RedEdge), B5(RedEdge)
        [9,10,8],   % Moisture: B11(SWIR1), B12(SWIR2), B8A(NIR-narrow)
        [10,10,6]   % Filler: B12, B12, B7
    };
    
    for t = 1:numel(triplets)
        bands = triplets{t};
        X_triplet = X(:,:,bands);
        
        % CRITICAL: Store original ranges BEFORE normalizing
        ch_ranges = zeros(3, 2);  % [min, max] for each channel
        X_normalized = zeros(size(X_triplet), 'single');
        
        for ch = 1:3
            ch_min = min(X_triplet(:,:,ch), [], 'all');
            ch_max = max(X_triplet(:,:,ch), [], 'all');
            ch_ranges(ch,:) = [ch_min, ch_max];  % STORE THESE!
            
            if ch_max > ch_min
                X_normalized(:,:,ch) = (X_triplet(:,:,ch) - ch_min) / (ch_max - ch_min);
            else
                X_normalized(:,:,ch) = 0.5;  % Middle gray for flat bands
            end
        end
        
        % Option 1: Keep as float (better precision)
        X_sr_triplet = ESRGAN_2xSuperResolution(single(X_normalized));
        
        % Option 2: If you must use uint8
        % X_sr_triplet = ESRGAN_2xSuperResolution(uint8(X_normalized * 255));
        
        % CRITICAL: Restore original ranges
        for i = 1:3
            bandIdx = bands(i);
            if bandIdx <= 10 && ~bandWritten(bandIdx)
                ch_min = ch_ranges(i, 1);
                ch_max = ch_ranges(i, 2);
                
                % Map [0,1] back to [ch_min, ch_max]
                X_sr(:,:,bandIdx) = X_sr_triplet(:,:,i) * (ch_max - ch_min) + ch_min;
                
                bandWritten(bandIdx) = true;
            end
        end
    end
end