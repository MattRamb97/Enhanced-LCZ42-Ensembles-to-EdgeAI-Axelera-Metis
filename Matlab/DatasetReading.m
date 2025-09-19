%download the dataset from
%https://dataserv.ub.tum.de/index.php/s/m1483140

clear all
warning off

%save training and test sets in .mat files

labelTR = h5read('training.h5','/label');
labelTR = uint8(labelTR);

TRset = h5read('training.h5','/sen2');
TRset = single(TRset);

labelTE = h5read('testing.h5','/label');
labelTE= uint8(labelTE);

TEset = h5read('testing.h5','/sen2');
TEset = single(TEset);

labelTV = h5read('validation.h5','/label');
labelTV = uint8(labelTV);

Vset = h5read('validation.h5','/sen2');
Vset = single(Vset);

save('LCZ.mat','labelTR','labelTE','labelTV','TRset','TEset','Vset');

%% SAR images
TEset_SAR = h5read('testing.h5','/sen1');
TEset_SAR = single(TEset_SAR);
TRset_SAR = h5read('training.h5','/sen1');
TRset_SAR = single(TRset_SAR);
save('LCZ_SAR.mat','TRset_SAR','TEset_SAR');