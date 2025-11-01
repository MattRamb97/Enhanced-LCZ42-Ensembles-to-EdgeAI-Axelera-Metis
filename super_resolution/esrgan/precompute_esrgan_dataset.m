function precompute_esrgan_dataset(inputH5, outputH5, scaleFactor)
    if isfile(outputH5)
        delete(outputH5);
    end
    
    info_sen2 = h5info(inputH5, '/sen2');
    N = info_sen2.Dataspace.Size(4);
    
    fprintf('Processing %d patches with ESRGAN %dx...\n', N, scaleFactor);
    
    % Create output with HALF precision (float16)
    newSize = [10, 32*scaleFactor, 32*scaleFactor, N];
    h5create(outputH5, '/sen2', newSize, 'Datatype', 'float16');  
    
    % Copy labels
    labels = h5read(inputH5, '/label');
    h5create(outputH5, '/label', size(labels), 'Datatype', 'uint8');
    h5write(outputH5, '/label', labels);
    
    % Process batches
    batchSize = 256;
    for i = 1:batchSize:N
        endIdx = min(i+batchSize-1, N);
        fprintf('  Batch %d-%d / %d\n', i, endIdx, N);
        
        X_batch = h5read(inputH5, '/sen2', [1 1 1 i], [10 32 32 endIdx-i+1]);
        X_sr_batch = zeros(10, 32*scaleFactor, 32*scaleFactor, endIdx-i+1, 'single');
        
        for j = 1:size(X_batch, 4)
            X = permute(X_batch(:,:,:,j), [2 3 1]);
            X = X * (255 / 2.8);
            
            X_sr = apply_esrgan_all_bands(X, scaleFactor);
            
            X_sr = X_sr * (2.8 / 255);
            X_sr_batch(:,:,:,j) = permute(X_sr, [3 1 2]);
        end
        
        % Convert to half precision before writing
        X_sr_batch_half = half(X_sr_batch);  
        h5write(outputH5, '/sen2', X_sr_batch_half, [1 1 1 i], size(X_sr_batch));
    end
    
    fprintf('Saved: %s (float16)\n', outputH5);
end