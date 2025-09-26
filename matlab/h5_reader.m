function X = h5_reader(pth, index, modality)
% Returns a single patch as HxWxB (single).
% pth: string to .h5  | index: 1-based sample index | modality: "MS" or "SAR"

switch modality
    case "MS"   % Sentinel-2: B2..B12 → 10 bands
        % H5 layout is (H, W, B, N). We read one sample along 4th dim.
        X = h5read(pth,'/sen2', [1 1 1 index], [32 32 10 1]);
    case "SAR"  % Sentinel-1: 8 bands (see dataset README)
        X = h5read(pth,'/sen1', [1 1 1 index], [32 32 8 1]);
    otherwise
        error('Unknown modality: %s', modality);
end
X = single(X);  % return single precision
end