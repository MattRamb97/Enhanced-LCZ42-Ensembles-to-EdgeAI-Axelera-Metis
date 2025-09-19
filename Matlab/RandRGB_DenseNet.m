clear all
warning off

gpuDevice(1) %running tests on GPU

%load data
load('LCZ.mat','labelTR','labelTE','labelTV','TRset','TEset','Vset');

siz=[224 224];%input size DenseNet
numClasses=17;%number of classes

%labels for training/testing the net
[a,labelTR]=max(labelTR);
[a,labelTE]=max(labelTE);

%image normalization
TRset=uint8(TRset./(2.8/255));
TEset=uint8(TEset./(2.8/255));


for reiteration=1:10%number of classifiers of the ensemble
    close all force

    %3 random channels
    [a,b]=sort(rand(10,1));
    chosen=b(1:3);

    %random RGB
    [a,b]=sort(rand(3,1));
    chosen(1)=b(1);%a random RGB band is selected 
    augTrainingImages=TRset(chosen,:,:,:);%replace the first channel with a RGB band
    augTestImages=TEset(chosen,:,:,:);%replace the first channel with a RGB band

    augTraining=zeros([224,224,3,size(augTrainingImages,4)],'uint8');
    augTest=zeros([224,224,3,size(augTestImages,4)],'uint8');

    %training set
    for pattern=1:length(augTraining)
        for ch=1:3
            IM(:,:,ch)=augTrainingImages(ch,:,:,pattern);
        end
        augTraining(:,:,:,pattern)=imresize(IM,siz);
    end

    %test set
    for pattern=1:length(augTest)
        for ch=1:3
            IM(:,:,ch)=augTestImages(ch,:,:,pattern);
        end
        augTest(:,:,:,pattern)=imresize(IM,siz);
    end

    %load the pretrained network
    netName='densenet201';
    net= eval(netName);
    %training options
    miniBatchSize = 50;
    learningRate = 1e-3;
    metodoOptim='sgdm';
    options = trainingOptions(metodoOptim,...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',7,...
        'InitialLearnRate',learningRate,...
        'Verbose',false,...
        'Plots','training-progress');

    %############training############
    %ReplaceLayersTransferLearning
    trainingImages = augmentedImageDatastore(siz,augTraining,categorical(labelTR));
    clear augTraining
    lgraph = layerGraph(net);
    lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    lgraph = connectLayers(lgraph,'avg_pool','fc');

    %training the net
    netTransfer = trainNetwork(trainingImages,lgraph,options);

    %test data classification
    [YPred,scores] = classify(netTransfer,augTest);


    %###########save scores in output file, e.g.#########
    outFileName = strcat("Z:\DATA\LCZ\SavedScores\LCZRandomOneRGB_", netName, '_',int2str(reiteration),".mat");
    save(outFileName, "scores","labelTE",'-v7.3');


end
