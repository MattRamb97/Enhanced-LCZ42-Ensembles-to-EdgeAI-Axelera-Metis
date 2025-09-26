function make_tables_from_h5(root)
% root = 'data/lcz42';
tr  = fullfile(root,'training.h5');
va  = fullfile(root,'validation.h5');
te  = fullfile(root,'testing.h5');

% Read sizes from H5 headers (no full load)
Ntr = double(h5info(tr,'/sen2').Dataspace.Size(end));   % samples
Nva = double(h5info(va,'/sen2').Dataspace.Size(end));
Nte = double(h5info(te,'/sen2').Dataspace.Size(end));

% Labels are one-hot in H5 → we store only the index 1..17
lab_tr = onehot_to_idx(h5read(tr,'/label')');  % N x 17 → N
lab_va = onehot_to_idx(h5read(va,'/label')');
lab_te = onehot_to_idx(h5read(te,'/label')');

% City IDs are not given in H5. We create a surrogate ID so that
% calibration split is random but **reproducible** within TRAIN.
rng(42,'twister');
city_tr = string(randi(50,[Ntr 1]));  % 50 pseudo cities for TRAIN
city_va = repmat("val",Nva,1);
city_te = repmat("test",Nte,1);

% Build tables for *MS* (Sentinel-2) and *SAR* (Sentinel-1) separately.
% Path column holds the .h5 filename; Index column points to the sample.
train_MS = table(repmat(string(tr),Ntr,1), lab_tr, city_tr, (1:Ntr)', repmat("MS",Ntr,1), ...
    'VariableNames',{'Path','Label','CityID','Index','Modality'});
val_MS   = table(repmat(string(va),Nva,1), lab_va, city_va, (1:Nva)', repmat("MS",Nva,1), ...
    'VariableNames',{'Path','Label','CityID','Index','Modality'});
test_MS  = table(repmat(string(te),Nte,1), lab_te, city_te, (1:Nte)', repmat("MS",Nte,1), ...
    'VariableNames',{'Path','Label','CityID','Index','Modality'});

train_SAR = train_MS; train_SAR.Modality(:) = "SAR";
val_SAR   = val_MS;   val_SAR.Modality(:)   = "SAR";
test_SAR  = test_MS;  test_SAR.Modality(:)  = "SAR";

% Save to disk
save(fullfile(root,'tables_MS.mat'),'train_MS','val_MS','test_MS','-v7.3');
save(fullfile(root,'tables_SAR.mat'),'train_SAR','val_SAR','test_SAR','-v7.3');

fprintf('Saved tables:\n  %s\n  %s\n', ...
    fullfile(root,'tables_MS.mat'), fullfile(root,'tables_SAR.mat'));
end

function idx = onehot_to_idx(M)
% M: N x 17 one-hot or probabilities
[~,idx] = max(M,[],2);
idx = categorical(idx);
end