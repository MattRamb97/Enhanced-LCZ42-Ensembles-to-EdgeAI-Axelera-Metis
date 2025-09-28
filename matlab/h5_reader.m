% Returns ONE patch as HxWxC (single) from So2Sat LCZ42 H5 files.
% Return one patch as [32 x 32 x C] single.
% Our files use: sen1=[8,32,32,N], sen2=[10,32,32,N] (index is 4th dim).
%
% Matteo Rambaldi — Thesis utilities

function X = h5_reader(pth, index, modality)

    arguments
        pth (1,1) string
        index (1,1) double {mustBeInteger, mustBePositive}
        modality (1,1) string
    end
    
    modality = upper(string(modality));
    switch modality
        case "MS"   % Sentinel-2 (10 bands)
            % start/count on [B,H,W,N]
            X4 = h5read(pth, '/sen2', [1 1 1 index], [10 32 32 1]);  % -> [10,32,32,1]
        case "SAR"  % Sentinel-1 (8 bands)
            X4 = h5read(pth, '/sen1', [1 1 1 index], [ 8 32 32 1]);  % -> [ 8,32,32,1]
        otherwise
            error('Unknown modality: %s', modality);
    end
    
    X = permute(X4, [2 3 1 4]);   % [32,32,C,1]
    X = single(X(:,:, :, 1));     % [32,32,C]

end

% NOTE 
% DenseNet expects 3 channels. We have here 10 (MS) and 8 (SAR). 
% The triplet selection (e.g., RGB = B4,B3,B2 for MS; SAR triplet 
% like [VV_lee, VH_lee, |C12|]) happens in the model training scripts 
% (Rand/RandRGB/SAR), just like professor's design. The reader returns all 
% bands so the training code can form the desired 3-ch view 
% deterministically or randomly.