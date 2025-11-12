% Returns ONE patch as HxWxC (single) from So2Sat LCZ42 H5 files.
%
% Matteo Rambaldi — Thesis utilities

function X = h5_reader(pth, index, modality)

    arguments
        pth (1,1) string
        index (1,1) double {mustBeInteger, mustBePositive}
        modality (1,1) string
    end

    modality = upper(modality);
    switch modality
        case "MS"
            dataset = '/sen2';
        case "SAR"
            dataset = '/sen1';
        otherwise
            error("Unknown modality: %s", modality);
    end

    info = h5info(pth, dataset);
    dims = info.Dataspace.Size;  % [C, H, W, N]

    C = dims(1);
    H = dims(2);
    W = dims(3);

    % Dynamically read the correct shape for 32x32 or 64x64
    X4 = h5read(pth, dataset, [1 1 1 index], [C H W 1]);
    X = permute(X4, [2 3 1 4]);   % [H W C 1] → [H W C]
    X = single(X(:,:, :, 1));
end