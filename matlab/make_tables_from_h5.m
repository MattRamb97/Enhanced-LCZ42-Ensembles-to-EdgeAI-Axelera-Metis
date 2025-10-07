% MAKE_TABLES_FROM_H5  Build light tables (MS/SAR) pointing into LCZ42 HDF5 files.
% root = 'data/lcz42'; with training.h5, validation.h5, testing.h5
%
% Produces:
%   tables_MS.mat  -> train_MS, val_MS, test_MS
%   tables_SAR.mat -> train_SAR, val_SAR, test_SAR
%
% Each table has variables: Path (string), Label (categorical 1..17),
% Index (1-based), Modality ("MS"/"SAR")
%
% Matteo Rambaldi — Thesis utilities

function make_tables_from_h5(root)

    arguments
        root (1,1) string
    end
    
    tr = fullfile(root,'training.h5');
    te = fullfile(root,'testing.h5');
    assert(isfile(tr) && isfile(te), 'Missing H5 files in %s', root);
    
    % --- read counts and labels (labels are one-hot N x 17) ---
    [Ntr, lab_tr] = getN_and_labels(tr);
    [Nte, lab_te] = getN_and_labels(te);
    
    % --- build Sentinel-2 (MS) tables ---
    train_MS = table(repmat(string(tr),Ntr,1), lab_tr, (1:Ntr)', repmat("MS",Ntr,1), ...
        'VariableNames',{'Path','Label','Index','Modality'});
    test_MS  = table(repmat(string(te),Nte,1), lab_te, (1:Nte)', repmat("MS",Nte,1), ...
        'VariableNames',{'Path','Label','Index','Modality'});
    
    % --- build Sentinel-1 (SAR) tables (same rows; Modality changes) ---
    train_SAR = train_MS; train_SAR.Modality(:) = "SAR";
    test_SAR  = test_MS;  test_SAR.Modality(:)  = "SAR";
    
    % --- save ---
    save(fullfile(root,'tables_MS.mat'),'train_MS','test_MS','-v7.3');
    save(fullfile(root,'tables_SAR.mat'),'train_SAR','test_SAR','-v7.3');
    
    fprintf('Saved:\n  %s\n  %s\n', fullfile(root,'tables_MS.mat'), fullfile(root,'tables_SAR.mat'));
end

% ======================================================================
%                               HELPERS
% ======================================================================

% Shapes of the files:
%   /sen1 : [8, 32, 32, N]
%   /sen2 : [10,32, 32, N]
%   /label: [17, N]
function [N, labels_cat] = getN_and_labels(h5path)
    d1 = h5info(h5path,'/sen1').Dataspace.Size;
    d2 = h5info(h5path,'/sen2').Dataspace.Size;
    dl = h5info(h5path,'/label').Dataspace.Size;
    
    assert(numel(d1)==4 && d1(1)==8  && d1(2)==32 && d1(3)==32, '/sen1 dims unexpected: %s', mat2str(d1));
    assert(numel(d2)==4 && d2(1)==10 && d2(2)==32 && d2(3)==32, '/sen2 dims unexpected: %s', mat2str(d2));
    assert(numel(dl)==2 && dl(1)==17, '/label dims unexpected: %s', mat2str(dl));
    
    N = d1(4);
    assert(d2(4)==N && dl(2)==N, 'N mismatch in %s (sen1/sen2/label).', h5path);
    
    L = h5read(h5path,'/label');   % [17, N]
    labels_cat = onehot_to_cat(L);
end

function C = onehot_to_cat(L)
    % L: [17, N]
    if size(L,1)==17
        [~,idx] = max(L, [], 1);  % 1×N
        idx = idx.';              % N×1
    else
        error('Unexpected label size: %s', mat2str(size(L)));
    end
    C = categorical(idx);
end