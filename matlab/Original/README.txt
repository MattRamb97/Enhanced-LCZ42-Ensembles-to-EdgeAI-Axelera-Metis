DatasetReading.m  creates .mat used in the classification functions
ensembleSARchannel_DenseNet.m  train the ensemble using SAR data
RandRGB_DenseNet.m train the RandRGB ensemble
Rand_DenseNet.m train the Rand ensemble



%%%%%%%%  fusion %%%%%%%%%%%%%%%%%%%%

%we apply the sum rule
clear all
warning off
cd('Z:\DATA\LCZ\SavedScores')%save all the scores in a given folder
reti=dir('Z:\DATA\LCZ\SavedScores\*.mat')%all the nets

SC=zeros(24188,17);
for qualeRete=1:length(reti)
    load(reti(qualeRete).name);
    SC=SC+scores;%sum rule
end

%performance of the ensemble
[a,b]=max(SC');
sum(b==labelTE)/length(labelTE) 