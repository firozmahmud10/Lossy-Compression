% paths to train and test folders
trainFolder0 =  'Path to train folder Uninfected';
trainFolder1 =  'Path to train folder infected';
testFolder0 = "Path to test folder Uninfected";
testFolder1 = "Path to test folder infected";

% Load and preprocess images 
images0 = dir(fullfile(trainFolder0, '*.png')); 
trainImages = {};
labels = [];
for i = 1:numel(images0)
    img = imread(fullfile(trainFolder0, images0(i).name));
    img = imresize(img, [50, 50]); 
    trainImages{end+1} = img;
    labels = [labels; 0]; 
end

images1 = dir(fullfile(trainFolder1, '*.png'));
for i = 1:numel(images1)
    img = imread(fullfile(trainFolder1, images1(i).name));
    img = imresize(img, [50, 50]);  
    trainImages{end+1} = img;
    labels = [labels; 1]; 
end

for i = 1:numel(trainImages)
    trainImages{i} = im2double(trainImages{i});
end

X = cat(4, trainImages{:});  
X = reshape(X, [], numel(trainImages)); 
hiddenSize1 = 1500;  
hiddenSize2 = 500; 
autoenc1 = trainAutoencoder(X, hiddenSize1, ...
    'MaxEpochs',1000, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
feat1 = encode(autoenc1, X);
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',1000, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feat2 = encode(autoenc2, feat1);

expected_num_samples = numel(images0) + numel(images1);
if size(feat2, 2) ~= expected_num_samples
    error('Number of samples in feat2 and labels do not match.');
end

labels = labels'; 
% softmax layer
softmaxnet = trainSoftmaxLayer(feat3, labels, 'MaxEpochs', 1000);
stackednet = stack(autoenc1,autoenc2,softmaxnet);
% Load and preprocess test images
testImages0 = dir(fullfile(testFolder0, '*.png'));
testImages = {};
testLabels = [];
testFilenames = {};
for i = 1:numel(testImages0)
    img = imread(fullfile(testFolder0, testImages0(i).name));
    img = imresize(img, [50, 50]);  
    testImages{end+1} = img;
    testLabels = [testLabels; 0];
     testFilenames{end+1} = testImages0(i).name; 
end

testImages1 = dir(fullfile(testFolder1, '*.png'));
for i = 1:numel(testImages1)
    img = imread(fullfile(testFolder1, testImages1(i).name));
    img = imresize(img, [50, 50]);  
    testImages{end+1} = img;
    testLabels = [testLabels; 1]; 
    testFilenames{end+1} = testImages1(i).name; 
end

for i = 1:numel(testImages)
    testImages{i} = im2double(testImages{i});
end

testX = cat(4, testImages{:});  
testX = reshape(testX, [], numel(testImages));  
predictedLabels = stackednet(testX);
% confusion matrix
plotconfusion(testLabels', predictedLabels);
%fine tune
stackednet = train(stackednet,X,labels);
predictedLabels = stackednet(testX);
predictedLabels = predictedLabels > 0.5;  

