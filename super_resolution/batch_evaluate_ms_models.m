% Complete evaluation script for MS super-resolution models
clear; clc;

fprintf('===== MS SUPER-RESOLUTION EVALUATION =====\n');
fprintf('System: MacBook Pro M4 Pro (14 cores)\n');
fprintf('Date: %s\n\n', datetime('now'));

% ========== CONFIGURATION ==========
% Update these paths to match your system
base_path = "../data/lcz42/";
output_path = "metrics/";

% Create output directory if it doesn't exist
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% MS-only models to evaluate
models = {
    'VDSRx2',       'training_vdsr2x.h5';        % correct
    'VDSRx3',       'training_vdsr3x.h5';        % correct
    'EDSRx2',       'training_edsr2x.h5';        % correct
    'EDSRx4',       'training_edsr4x.h5';        % correct
    'ESRGANx2',     'training_esrgan2x.h5';      % correct
    'SWINIRx2',     'training_swinir2x.h5';      % correct
    'BSRNETx2',     'training_bsrnet2x.h5';      % correct
    'RealESRGANx4', 'training_realesrgan4x.h5'   % correct
};

% ========== SETUP ==========
% Original HR file
h5_orig = fullfile(base_path, "training.h5");

% Check if original file exists
if ~isfile(h5_orig)
    error('Original HR file not found: %s', h5_orig);
end

% Get actual number of samples from the file
info = h5info(h5_orig, '/sen2');
total_samples = info.Dataspace.Size(4); % 4th dimension in MATLAB is N
fprintf('Total samples in dataset: %d\n', total_samples);
fprintf('HR data dimensions (MATLAB): %dx%dx%dx%d\n', info.Dataspace.Size);
fprintf('(bands x height x width x samples)\n\n');

% Use all samples or specify a limit for testing
num_samples = total_samples;  % Use all samples

% Initialize results storage
num_models = size(models, 1);
resultsData = struct();

% Time estimation (based on ~300 samples/sec from your test)
est_time_per_sample = 1/300;
est_time_per_model = num_samples * est_time_per_sample / 60; % minutes
fprintf('Samples to process per model: %d\n', num_samples);
fprintf('Estimated time per model: %.1f minutes\n', est_time_per_model);
fprintf('Estimated total time: %.1f hours\n\n', num_models * est_time_per_model / 60);

% ========== MAIN EVALUATION LOOP ==========
total_start = tic;

for i = 1:num_models
    model_name = models{i, 1};
    model_file = models{i, 2};
    h5_sr = fullfile(base_path, model_file);
    
    fprintf('\n╔══════════════════════════════════════════╗\n');
    fprintf('║  Model %d/%d: %-28s║\n', i, num_models, model_name);
    fprintf('╚══════════════════════════════════════════╝\n');
    
    % Check if SR file exists
    if ~isfile(h5_sr)
        warning('File not found: %s', h5_sr);
        resultsData.(model_name) = struct('MeanPSNR', NaN, 'MeanSSIM', NaN, 'MeanRMSE', NaN);
        continue;
    end
    
    % Verify SR file dimensions
    try
        sr_info = h5info(h5_sr, '/sen2');
        fprintf('SR dimensions: %dx%dx%dx%d\n', sr_info.Dataspace.Size);
        
        % Check if sample counts match
        if sr_info.Dataspace.Size(4) ~= info.Dataspace.Size(4)
            warning('Sample count mismatch! HR: %d, SR: %d', ...
                info.Dataspace.Size(4), sr_info.Dataspace.Size(4));
        end
    catch
        warning('Cannot read SR file info');
    end
    
    % Evaluate model
    try
        model_start = tic;
        
        % Call the parallel evaluation function
        metrics = evaluate_sr_metrics_dataset_parallel(h5_orig, h5_sr, num_samples);
        
        model_time = toc(model_start);
        
        % Store detailed metrics
        resultsData.(model_name) = metrics;
        
        fprintf('\n✓ %s completed in %.1f minutes\n', model_name, model_time/60);
        
        % Save checkpoint after each model (in case of crash)
        checkpoint_file = fullfile(output_path, sprintf('checkpoint_ms_%s.mat', model_name));
        save(checkpoint_file, 'metrics');
        fprintf('  Checkpoint saved: %s\n', checkpoint_file);
        
    catch ME
        fprintf('\n✗ ERROR in %s:\n', model_name);
        fprintf('  %s\n', ME.message);
        fprintf('  Line %d in %s\n', ME.stack(1).line, ME.stack(1).name);
        resultsData.(model_name) = struct('MeanPSNR', NaN, 'MeanSSIM', NaN, 'MeanRMSE', NaN);
    end
    
    % Progress summary
    elapsed = toc(total_start);
    if i < num_models
        eta = (elapsed/i) * (num_models-i);
        fprintf('\nOverall progress: %d/%d models completed\n', i, num_models);
        fprintf('Time elapsed: %.1f min | ETA: %.1f min\n', elapsed/60, eta/60);
    end
end

% ========== CREATE RESULTS TABLE ==========
fprintf('\n\nCreating results table...\n');

model_names = fieldnames(resultsData);
num_results = length(model_names);

Method = strings(num_results, 1);
SSIM = zeros(num_results, 1);
RMSE = zeros(num_results, 1);
PSNR_dB = zeros(num_results, 1);

for i = 1:num_results
    name = model_names{i};
    Method(i) = name;
    
    if isfield(resultsData.(name), 'MeanSSIM')
        SSIM(i) = resultsData.(name).MeanSSIM;
        RMSE(i) = resultsData.(name).MeanRMSE;
        PSNR_dB(i) = resultsData.(name).MeanPSNR;
    else
        SSIM(i) = NaN;
        RMSE(i) = NaN;
        PSNR_dB(i) = NaN;
    end
end

% Create and format table
resultsTable = table(Method, SSIM, RMSE, PSNR_dB);
resultsTable.SSIM = round(resultsTable.SSIM, 4);
resultsTable.RMSE = round(resultsTable.RMSE, 4);
resultsTable.PSNR_dB = round(resultsTable.PSNR_dB, 2);

% Sort by PSNR (best first)
resultsTable = sortrows(resultsTable, {'PSNR_dB','SSIM'}, {'descend','descend'});

% ========== DISPLAY FINAL RESULTS ==========
fprintf('\n\n╔══════════════════════════════════════════╗\n');
fprintf('║      MS EVALUATION RESULTS TABLE         ║\n');
fprintf('╚══════════════════════════════════════════╝\n\n');
disp(resultsTable);

% ========== SAVE RESULTS ==========
final_file = fullfile(output_path, 'ms_metrics_training.mat');
save(final_file, 'resultsTable', 'resultsData');
fprintf('\nFinal results saved to: %s\n', final_file);

% ========== GENERATE LATEX TABLE ==========
latex_file = fullfile(output_path, 'results_table.tex');
fid = fopen(latex_file, 'w');
fprintf(fid, '\\begin{table}[h]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Super-Resolution Performance on LCZ42 MS Dataset}\n');
fprintf(fid, '\\label{tab:sr_results}\n');
fprintf(fid, '\\begin{tabular}{lccc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Method & SSIM & RMSE & PSNR (dB) \\\\\n');
fprintf(fid, '\\midrule\n');
for i = 1:height(resultsTable)
    fprintf(fid, '%s & %.4f & %.4f & %.2f \\\\\n', ...
        resultsTable.Method(i), resultsTable.SSIM(i), ...
        resultsTable.RMSE(i), resultsTable.PSNR_dB(i));
end
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('LaTeX table saved to: %s\n', latex_file);

% ========== FINAL SUMMARY ==========
total_time = toc(total_start);
fprintf('\n═══════════════════════════════════════════\n');
fprintf('  MS EVALUATION COMPLETE!\n');
fprintf('═══════════════════════════════════════════\n');
fprintf('Total samples processed: %d x %d models\n', num_samples, num_models);
fprintf('Total time: %.1f hours (%.1f min/model)\n', total_time/3600, total_time/60/num_models);
fprintf('Results saved to: %s\n', final_file);

% ========== VALIDATION CHECK ==========
fprintf('\n--- Validation Check ---\n');
num_valid = sum(~isnan(resultsTable.PSNR_dB));
fprintf('Successfully evaluated: %d/%d models\n', num_valid, num_models);

if num_valid > 0
    fprintf('\nBest performing model:\n');
    fprintf('  %s: PSNR=%.2f dB, SSIM=%.4f, RMSE=%.4f\n', ...
        resultsTable.Method(1), resultsTable.PSNR_dB(1), ...
        resultsTable.SSIM(1), resultsTable.RMSE(1));
end

fprintf('\n✓ Ready for Zenodo upload!\n');