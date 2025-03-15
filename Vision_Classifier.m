
net = visionTransformer;
inputSize = net.Layers(1).InputSize; 

net = freezeNetwork(net, 'LayersToIgnore', "SelfAttentionLayer");
imageFolder = "Path to Images";


% Create image datastore and split data into training, validation, and test sets
imds = imageDatastore(imageFolder, IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);

% Split
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.75, 0.2);

% Data augmentation 
augmenter = imageDataAugmenter( ...
    RandXReflection = true, ...
    RandRotation = [-90 90], ...
    RandScale = [1 2]);


augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, DataAugmentation = augmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

analyzeNetwork(net)

layer = fullyConnectedLayer(numClasses, Name = "head");
net = replaceLayer(net, "head", layer);

% training options
miniBatchSize =24; 
numObservationsTrain = numel(augimdsTrain.Files);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);

options = trainingOptions("adam", ...
    MaxEpochs = 3, ...
    InitialLearnRate = 0.0005, ...
    MiniBatchSize = miniBatchSize, ...
    ValidationData = augimdsValidation, ...
    ValidationFrequency = numIterationsPerEpoch, ...
    OutputNetwork = "best-validation", ...
    Plots = "training-progress", ...
    Metrics = "accuracy", ...
    ExecutionEnvironment = "Multi-GPU", ...  
    Verbose = false);

gpuDevice(1);


net = trainnet(augimdsTrain, net, "crossentropy", options);

YPred = minibatchpredict(net, augimdsTest);


YPred = onehotdecode(YPred, classNames, 2);

TTest = imdsTest.Labels;
confusionchart(TTest, YPred);

accuracy = mean(YPred == TTest);
disp("Test Accuracy: " + accuracy);
