function EnableGPU(idx)
    try
        if gpuDeviceCount("available") >= 1
            if nargin < 1
                idx = 1;
            end
            g = gpuDevice(idx);
            fprintf('Using GPU %d: %s\n', g.Index, g.Name);
        else
            fprintf('No CUDA GPU detected. Using CPU.\n');
        end
    catch ME
        warning('EnableGPU:InitFailed', ...
            'GPU init failed: %s. Continuing on CPU.', ME.message);
    end
end