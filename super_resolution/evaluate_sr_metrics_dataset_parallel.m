function metrics = evaluate_sr_metrics_dataset_parallel(h5_orig, h5_sr, num_samples)
    % Parallel optimized evaluation for M4 Pro
    % modality: "MS" for /sen2 only
    % num_samples: optional, set to limit samples (e.g., for testing)
    
    if nargin < 4
        num_samples = inf; % Process all samples
    end
    
    % For MS only (since SR methods are only for /sen2)
    datasetName = '/sen2';
    numBands = 10;
    max_value = 2.8;
    
    % Get actual dimensions from the file
    info_orig = h5info(h5_orig, datasetName);
    info_sr = h5info(h5_sr, datasetName);
    
    % MATLAB reads Nx32x32x10 as 10x32x32xN
    N_orig = info_orig.Dataspace.Size(4);
    N_sr = info_sr.Dataspace.Size(4);
    
    % Verify both files have same number of samples
    if N_orig ~= N_sr
        error('Mismatch in number of samples: HR has %d, SR has %d', N_orig, N_sr);
    end
    
    % Use the actual N or the specified limit
    N = min(N_orig, num_samples);
    
    fprintf('Dataset info:\n');
    fprintf('  Total samples in file: %d\n', N_orig);
    fprintf('  Samples to evaluate: %d\n', N);
    fprintf('  HR dimensions (MATLAB): %dx%dx%dx%d\n', info_orig.Dataspace.Size);
    fprintf('  SR dimensions (MATLAB): %dx%dx%dx%d\n', info_sr.Dataspace.Size);
    fprintf('  Evaluating MS (/sen2) data only\n\n');
    
    % Pre-allocate arrays
    psnr_vals = zeros(N, numBands);
    ssim_vals = zeros(N, numBands);
    rmse_vals = zeros(N, numBands);
    
    % Batch processing parameters
    batch_size = 1024;
    num_batches = ceil(N / batch_size);
    
    % Check for Parallel Computing Toolbox
    has_parallel = license('test', 'Distrib_Computing_Toolbox');
    
    if has_parallel
        % M4 Pro: 10 performance + 4 efficiency cores
        num_workers = min(12, maxNumCompThreads);
        
        current_pool = gcp('nocreate');
        if isempty(current_pool) || current_pool.NumWorkers ~= num_workers
            delete(current_pool);
            parpool('threads', num_workers);
        end
        fprintf('Using parallel processing with %d workers\n', num_workers);
    else
        fprintf('Processing sequentially\n');
    end
    
    % Process in batches
    tic;
    for batch = 1:num_batches
        start_idx = (batch - 1) * batch_size + 1;
        end_idx = min(batch * batch_size, N);
        batch_indices = start_idx:end_idx;
        current_batch_size = length(batch_indices);
        
        % Pre-load batch data
        batch_hr = cell(current_batch_size, 1);
        batch_sr = cell(current_batch_size, 1);
        
        for j = 1:current_batch_size
            idx = batch_indices(j);
            % Read using the MS-specific reader
            batch_hr{j} = double(h5_reader_ms(h5_orig, idx));
            batch_sr{j} = double(h5_reader_ms(h5_sr, idx));
        end
        
        % Process batch
        batch_psnr = zeros(current_batch_size, numBands);
        batch_ssim = zeros(current_batch_size, numBands);
        batch_rmse = zeros(current_batch_size, numBands);
        
        if has_parallel
            parfor j = 1:current_batch_size
                [batch_psnr(j,:), batch_ssim(j,:), batch_rmse(j,:)] = ...
                    process_single_sample(batch_hr{j}, batch_sr{j}, numBands, max_value);
            end
        else
            for j = 1:current_batch_size
                [batch_psnr(j,:), batch_ssim(j,:), batch_rmse(j,:)] = ...
                    process_single_sample(batch_hr{j}, batch_sr{j}, numBands, max_value);
            end
        end
        
        % Store results
        psnr_vals(batch_indices, :) = batch_psnr;
        ssim_vals(batch_indices, :) = batch_ssim;
        rmse_vals(batch_indices, :) = batch_rmse;
        
        % Progress update
        if mod(batch, 10) == 0 || batch == num_batches
            elapsed = toc;
            samples_done = end_idx;
            rate = samples_done / elapsed;
            eta = (N - samples_done) / rate;
            fprintf('Progress: %d/%d samples (%.1f%%) - Rate: %.0f samples/sec - ETA: %.1f min\n', ...
                samples_done, N, 100*samples_done/N, rate, eta/60);
        end
    end
    
    total_time = toc;
    fprintf('Processing completed in %.1f minutes (%.0f samples/sec)\n', ...
        total_time/60, N/total_time);
    
    % Calculate statistics
    metrics.PSNR = psnr_vals;
    metrics.SSIM = ssim_vals;
    metrics.RMSE = rmse_vals;
    
    metrics.MeanPSNR = mean(psnr_vals(:), 'omitnan');
    metrics.MeanSSIM = mean(ssim_vals(:), 'omitnan');
    metrics.MeanRMSE = mean(rmse_vals(:), 'omitnan');
    
    metrics.StdPSNR = std(psnr_vals(:), 'omitnan');
    metrics.StdSSIM = std(ssim_vals(:), 'omitnan');
    metrics.StdRMSE = std(rmse_vals(:), 'omitnan');
    
    metrics.MeanPSNR_perBand = mean(psnr_vals, 1, 'omitnan');
    metrics.MeanSSIM_perBand = mean(ssim_vals, 1, 'omitnan');
    metrics.MeanRMSE_perBand = mean(rmse_vals, 1, 'omitnan');
    
    % Display summary
    fprintf('\n=== MS Evaluation Summary ===\n');
    fprintf('PSNR: %.2f ± %.2f dB\n', metrics.MeanPSNR, metrics.StdPSNR);
    fprintf('SSIM: %.4f ± %.4f\n', metrics.MeanSSIM, metrics.StdSSIM);
    fprintf('RMSE: %.4f ± %.4f\n', metrics.MeanRMSE, metrics.StdRMSE);
    fprintf('=============================\n');
end

function data = h5_reader_ms(filename, idx)
    % Read MS data from /sen2 dataset
    % Original: Nx32x32x10 → MATLAB: 10x32x32xN
    
    datasetName = '/sen2';
    
    % Read specific sample
    % MATLAB uses [bands, height, width, sample_index]
    start = [1, 1, 1, idx];
    count = [10, 32, 32, 1];  % Read all 10 bands, 32x32 patch, 1 sample
    
    data = h5read(filename, datasetName, start, count);
    
    % Reshape from 10x32x32x1 to 32x32x10
    data = permute(squeeze(data), [2, 3, 1]);
end

function [psnr_vals, ssim_vals, rmse_vals] = process_single_sample(X_hr, X_sr, numBands, max_value)
    % Process a single sample - optimized for parallel execution
    
    % Resize HR to match SR size
    X_hr = imresize(X_hr, [size(X_sr,1), size(X_sr,2)], 'bicubic');
    
    % Normalize to [0,1]
    if max(X_hr(:)) > 10
        X_hr = X_hr / 255;
        X_sr = X_sr / 255;
    elseif max(X_hr(:)) > 1.1
        X_hr = X_hr / max_value;
        X_sr = X_sr / max_value;
    end
    
    % Clip to valid range
    X_hr = max(0, min(1, X_hr));
    X_sr = max(0, min(1, X_sr));
    
    % Calculate metrics for each band
    psnr_vals = zeros(1, numBands);
    ssim_vals = zeros(1, numBands);
    rmse_vals = zeros(1, numBands);
    
    for c = 1:numBands
        hr_band = X_hr(:,:,c);
        sr_band = X_sr(:,:,c);
        
        psnr_vals(c) = psnr(sr_band, hr_band, 1);
        ssim_vals(c) = ssim(sr_band, hr_band);
        rmse_vals(c) = sqrt(mean((sr_band(:) - hr_band(:)).^2));
    end
end