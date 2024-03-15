mainFolderPath = 'C:\Users\Aravind\Desktop\ROE\trimester 2\Application for data science\Coursework 1\archive\chest_xray';

classes = {'train', 'test'};
subfolders = {'NORMAL', 'PNEUMONIA'};
trainData = cell(0,1); % Initialize an empty cell array for training data
testData = cell(0,1); % Initialize an empty cell array for testing data
trainLabels = [];
testLabels = [];
for i = 1:length(classes)
    class = classes{i};
    for j = 1:length(subfolders)
        subfolder = subfolders{j};
        currentFolderPath = fullfile(mainFolderPath, class, subfolder);
        files = dir(fullfile(currentFolderPath, '*.jpeg')); % Change the extension as needed
        for k = 1:length(files)
            currentFilePath = fullfile(currentFolderPath, files(k).name);
% Reading and processing image files
% Read the image file
            currentData = imread(currentFilePath);
            targetSize = [100 100];
            currentData = imresize(currentData, targetSize);
            if strcmp(class, 'train')
                trainData{end+1} = currentData;
                trainLabels = [trainLabels; j];
            elseif strcmp(class, 'test')
                testData{end+1} = currentData;
                testLabels = [testLabels; j];
            end
        end
    end
end
trainLabels = categorical(trainLabels, [1 2], {'normal', 'pneumonia'});
testLabels = categorical(testLabels, [1 2], {'normal', 'pneumonia'});
figure;
for i = 1:10
end
% Create a subplot for visualization
    subplot(2, 5, i);
% Display an image
    imshow(trainData{i});
% Add a title to the plot
    title([char(trainLabels(i))]);
startIndex = length(trainData) - 9; 
for i = startIndex:length(trainData)
end
% Create a subplot for visualization
    subplot(2, 5, i - startIndex + 1); 
% Display an image
    imshow(testData{i});
% Add a title to the plot
    title([char(testLabels(i))]);
startIndex = length(testData) - 9; 
for i = startIndex:length(testData)
end
disp(['Number of training samples: ' num2str(numel(trainData))]);
disp(['Number of testing samples: ' num2str(numel(testData))]);
disp(['Number of normal cases in training set: ' num2str(sum(trainLabels == 'normal'))]);
disp(['Number of pneumonia cases in training set: ' num2str(sum(trainLabels == 'pneumonia'))]);
disp(['Number of normal cases in testing set: ' num2str(sum(testLabels == 'normal'))]);
disp(['Number of pneumonia cases in testing set: ' num2str(sum(testLabels == 'pneumonia'))]);
% Create a subplot for visualization
subplot(2, 1, 1);
bar(categorical({'Normal', 'Pneumonia'}), [sum(trainLabels == 'normal'), sum(trainLabels == 'pneumonia')]);
% Add a title to the plot
title('Training Set - Class Distribution');
ylabel('Count');
% Create a subplot for visualization
subplot(2, 1, 2);
bar(categorical({'Normal', 'Pneumonia'}), [sum(testLabels == 'normal'), sum(testLabels == 'pneumonia')]);
% Add a title to the plot
title('Testing Set - Class Distribution');
for i = 1:length(trainData)
end
    if size(trainData{i}, 3) == 3
end
        trainData{i} = rgb2gray(trainData{i});
targetSize = [100 100];
augmenter = imageDataAugmenter( ...
    'RandRotation', [-30, 30], ... % Random rotations between -30 and 30 degrees
    'RandXReflection', true, ...   % Random horizontal flips
    'RandYReflection', true, ...   % Random vertical flips
    'RandScale', [0.8, 1.2]);      % Random scaling between 80% and 120%
minorityIndices = find(trainLabels == 'normal');
majorityIndices = find(trainLabels == 'pneumonia');
augmentAmount = length(majorityIndices) - length(minorityIndices);
augmentedData = cell(augmentAmount, 1);
augmentedLabels = repmat(categorical({'normal'}), augmentAmount, 1);
for i = 1:augmentAmount
end
    index = minorityIndices(randi(length(minorityIndices)));
    originalImage = trainData{index};
    augmentedImage = augment(augmenter, originalImage);
    resizedAugmentedImage = imresize(augmentedImage, targetSize);
    augmentedData{i} = resizedAugmentedImage;
balancedTrainData = trainData;
for i = 1:length(augmentedData)
end
    balancedTrainData{end+1} = augmentedData{i};
balancedTrainLabels = [trainLabels; augmentedLabels];
bar(categorical({'Normal', 'Pneumonia'}), [sum(balancedTrainLabels == 'normal'), sum(balancedTrainLabels == 'pneumonia')]);
if ndims(balancedTrainData{1}) == 3
end
    for i = 1:numel(balancedTrainData)
end
        balancedTrainData{i} = rgb2gray(balancedTrainData{i});
pixel_values_normal = [];
pixel_values_pneumonia = [];
for i = 1:numel(balancedTrainData)
end
    if balancedTrainLabels(i) == 'normal'
end
        pixel_values_normal = [pixel_values_normal; double(reshape(balancedTrainData{i}, [], 1))];
    elseif balancedTrainLabels(i) == 'pneumonia'
        pixel_values_pneumonia = [pixel_values_pneumonia; double(reshape(balancedTrainData{i}, [], 1))];
mean_pixel_value_normal = mean(pixel_values_normal);
mean_pixel_value_pneumonia = mean(pixel_values_pneumonia);
std_dev_pixel_value_normal = std(pixel_values_normal);
std_dev_pixel_value_pneumonia = std(pixel_values_pneumonia);
disp(['Mean Pixel Value (Normal): ', num2str(mean_pixel_value_normal)]);
disp(['Mean Pixel Value (Pneumonia): ', num2str(mean_pixel_value_pneumonia)]);
disp(['Standard Deviation Pixel Value (Normal): ', num2str(std_dev_pixel_value_normal)]);
disp(['Standard Deviation Pixel Value (Pneumonia): ', num2str(std_dev_pixel_value_pneumonia)]);
[h, p, ci, stats] = ttest2(pixel_values_normal, pixel_values_pneumonia);
disp(['t-statistic: ', num2str(stats.tstat)]);
disp(['p-value: ', num2str(p)]);
disp(['95% Confidence Interval: [', num2str(ci(1)), ', ', num2str(ci(2)), ']']);
if p < 0.05
end
    disp('There is a significant difference in mean pixel values between normal and pneumonia cases.');
else
    disp('There is no significant difference in mean pixel values between normal and pneumonia cases.');
pd_normal = fitdist(pixel_values_normal, 'Normal');
pd_pneumonia = fitdist(pixel_values_pneumonia, 'Normal');
x_values = linspace(min([pixel_values_normal; pixel_values_pneumonia]), max([pixel_values_normal; pixel_values_pneumonia]), 1000);
plot(x_values, pdf(pd_normal, x_values), 'b', 'LineWidth', 2);
hold on;
plot(x_values, pdf(pd_pneumonia, x_values), 'r', 'LineWidth', 2);
xlabel('Pixel Values');
ylabel('Probability Density');
% Add a title to the plot
title('Probability Distribution of Pixel Values');
legend('Normal Cases', 'Pneumonia Cases');
p_value = chi2gof([sum(balancedTrainLabels == 'normal'), sum(balancedTrainLabels == 'pneumonia')]);
disp(['Class Imbalance Chi-Squared Test: p-value = ', num2str(p_value)]);
t_test_result = ttest2(pixel_values_normal, pixel_values_pneumonia);
disp(['Pixel Value T-Test: p-value = ', num2str(t_test_result)]);
numTrain = numel(balancedTrainData);
numTest = numel(testData);
XTrain = zeros(100 * 100, numTrain);
for i = 1:numTrain
end
    if size(balancedTrainData{i}, 3) == 1
        grayImage = balancedTrainData{i};
    else
        grayImage = rgb2gray(balancedTrainData{i});
    end
    XTrain(:, i) = double(reshape(grayImage, [], 1)) / 255; % Normalize pixel values to [0, 1]
XTest = zeros(100 * 100, numTest);
for i = 1:numTest
end
    if size(testData{i}, 3) == 1
end
        grayImage = testData{i};
        grayImage = rgb2gray(testData{i});
    XTest(:, i) = double(reshape(grayImage, [], 1)) / 255;
coeff = pca(XTrain');
reduced_data = XTrain' * coeff(:, 1:2);
gscatter(reduced_data(:, 1), reduced_data(:, 2), balancedTrainLabels);
% Add a title to the plot
title('PCA: 2D Projection of Features');
binRange = 1:20;
numTreesRange = 1:4;
numNeighborsRange = 1:10;
bestSingleModelWithHOG = [];
bestSingleModelWithHOGAccuracy = 0;
bestSingleModelWithHOGName = '';
bestCombinedModelWithHOG = [];
bestCombinedModelWithHOGAccuracy = 0;
bestCombinedModelWithHOGName = '';
bestSingleModelWithoutHOG = [];
bestSingleModelWithoutHOGAccuracy = 0;
bestSingleModelWithoutHOGName = '';
bestCombinedModelWithoutHOG = [];
bestCombinedModelWithoutHOGAccuracy = 0;
bestCombinedModelWithoutHOGName = '';
bestCombinationWithHOGDetails = struct('NumTrees', 0, 'NumNeighbors', 0);
bestCombinationWithoutHOGDetails = struct('NumTrees', 0, 'NumNeighbors', 0);
for binIdx = 1:length(binRange)
end
    numBins = binRange(binIdx);
    hogFeaturesTrain = zeros(numel(balancedTrainData), length(extractHOGFeaturesFromImage(balancedTrainData{1}, numBins)));
        hogFeaturesTrain(i, :) = extractHOGFeaturesFromImage(balancedTrainData{i}, numBins);
    hogFeaturesTest = zeros(numel(testData), length(extractHOGFeaturesFromImage(testData{1}, numBins)));
    for i = 1:numel(testData)
end
        hogFeaturesTest(i, :) = extractHOGFeaturesFromImage(testData{i}, numBins);
    for treeIdx = 1:length(numTreesRange)
end
        numTrees = numTreesRange(treeIdx);
        randomForestModelWithHOG = TreeBagger(numTrees, hogFeaturesTrain, char(balancedTrainLabels), 'Method', 'classification');
        predictionsRFWithHOG = predict(randomForestModelWithHOG, hogFeaturesTest);
        predictionsRFWithHOG = categorical(predictionsRFWithHOG, {'1', '2'}, {'normal', 'pneumonia'});
        accuracyRFWithHOG = sum(predictionsRFWithHOG == testLabels) / numel(testLabels);
        disp(['Random Forest Accuracy with HOG with ', num2str(numTrees), ' trees: ', num2str(accuracyRFWithHOG)]);
        if accuracyRFWithHOG > bestSingleModelWithHOGAccuracy
end
            bestSingleModelWithHOGAccuracy = accuracyRFWithHOG;
            bestSingleModelWithHOG = randomForestModelWithHOG;
            bestSingleModelWithHOGName = 'Random Forest with HOG';
        svmModelWithHOG = fitcecoc(hogFeaturesTrain, balancedTrainLabels);
        predictionsSVMWithHOG = predict(svmModelWithHOG, hogFeaturesTest);
        accuracySVMWithHOG = sum(predictionsSVMWithHOG == testLabels) / numel(testLabels);
        disp(['SVM Accuracy with HOG: ', num2str(accuracySVMWithHOG)]);
        if accuracySVMWithHOG > bestSingleModelWithHOGAccuracy
end
            bestSingleModelWithHOGAccuracy = accuracySVMWithHOG;
            bestSingleModelWithHOG = svmModelWithHOG;
            bestSingleModelWithHOGName = 'SVM with hog';
        naiveBayesModelWithHOG = fitcnb((hogFeaturesTrain+1e-5), balancedTrainLabels);
        predictionsNBWithHOG = predict(naiveBayesModelWithHOG, hogFeaturesTest);
        accuracyNBWithHOG = sum(predictionsNBWithHOG == testLabels) / numel(testLabels);
        disp(['Naive Bayes Accuracy with HOG: ', num2str(accuracyNBWithHOG)]);
        if accuracyNBWithHOG > bestSingleModelWithHOGAccuracy
end
            bestSingleModelWithHOGAccuracy = accuracyNBWithHOG;
            bestSingleModelWithHOG = naiveBayesModelWithHOG;
            bestSingleModelWithHOGName = 'NB with hog';
        for neighborIdx = 1:length(numNeighborsRange)
end
            numNeighbors = numNeighborsRange(neighborIdx);
            knnModelWithHOG = fitcknn(hogFeaturesTrain, balancedTrainLabels, 'NumNeighbors', numNeighbors);
            predictionsKNNWithHOG = predict(knnModelWithHOG, hogFeaturesTest);
            accuracyKNNWithHOG = sum(predictionsKNNWithHOG == testLabels) / numel(testLabels);
            disp(['kNN Accuracy with HOG with ', num2str(numNeighbors), ' neighbors: ', num2str(accuracyKNNWithHOG)]);
            if accuracyKNNWithHOG > bestSingleModelWithHOGAccuracy
end
                bestSingleModelWithHOGAccuracy = accuracyKNNWithHOG;
                bestSingleModelWithHOG = knnModelWithHOG;
                bestSingleModelWithHOGName = 'knn with hog';
            if iscell(predictionsRFWithHOG)
end
                predictionsRFWithHOG = cellfun(@str2double, predictionsRFWithHOG);
            predictionsSVMWithHOG = double(predictionsSVMWithHOG);
            predictionsRFWithHOG = double(predictionsRFWithHOG);
            predictionsNBWithHOG = double(predictionsNBWithHOG);
            predictionsKNNWithHOG = double(predictionsKNNWithHOG);
            combinedPredictionsWithHOG = mode(cat(2, double(predictionsSVMWithHOG), predictionsRFWithHOG, double(predictionsNBWithHOG), double(predictionsKNNWithHOG)), 2);
            finalPredictionsWithHOG = arrayfun(@(x) categorical(x, [1 2], {'normal', 'pneumonia'}), combinedPredictionsWithHOG);
            ensembleAccuracyWithHOG = sum(finalPredictionsWithHOG == testLabels) / numel(testLabels);
            disp(['Ensemble Accuracy with HOG with ' num2str(numTrees) ' trees, ' num2str(numNeighbors), ' neighbors: ', num2str(ensembleAccuracyWithHOG)]);
            if ensembleAccuracyWithHOG > bestCombinedModelWithHOGAccuracy
end
                bestCombinedModelWithHOGAccuracy = ensembleAccuracyWithHOG;
                bestCombinedModelWithHOG = struct('SVM', svmModelWithHOG, 'RF', randomForestModelWithHOG, 'NB', naiveBayesModelWithHOG, 'kNN', knnModelWithHOG);
        randomForestModelWithoutHOG = TreeBagger(numTrees, XTrain', char(balancedTrainLabels), 'Method', 'classification');
        predictionsRFWithoutHOG = predict(randomForestModelWithoutHOG, XTest');
        predictionsRFWithoutHOG = categorical(predictionsRFWithoutHOG, {'1', '2'}, {'normal', 'pneumonia'});
        accuracyRFWithoutHOG = sum(predictionsRFWithoutHOG == testLabels) / numel(testLabels);
        disp(['Random Forest Accuracy without HOG with ', num2str(numTrees), ' trees: ', num2str(accuracyRFWithoutHOG)]);
        if accuracyRFWithoutHOG > bestSingleModelWithoutHOGAccuracy
end
            bestSingleModelWithoutHOGAccuracy = accuracyRFWithoutHOG;
            bestSingleModelWithoutHOG = randomForestModelWithoutHOG;
            bestSingleModelWithoutHOGName = 'rf without hog';
        svmModelWithoutHOG = fitcecoc(XTrain', balancedTrainLabels);
        predictionsSVMWithoutHOG = predict(svmModelWithoutHOG, XTest');
        accuracySVMWithoutHOG = sum(predictionsSVMWithoutHOG == testLabels) / numel(testLabels);
        disp(['SVM Accuracy without HOG: ', num2str(accuracySVMWithoutHOG)]);
        if accuracySVMWithoutHOG > bestSingleModelWithoutHOGAccuracy
end
            bestSingleModelWithoutHOGAccuracy = accuracySVMWithoutHOG;
            bestSingleModelWithoutHOG = svmModelWithoutHOG;
            bestSingleModelWithoutHOGName = 'svm without hog';
        naiveBayesModelWithoutHOG = fitcnb(XTrain', balancedTrainLabels);
        predictionsNBWithoutHOG = predict(naiveBayesModelWithoutHOG, XTest');
        accuracyNBWithoutHOG = sum(predictionsNBWithoutHOG == testLabels) / numel(testLabels);
        disp(['Naive Bayes Accuracy without HOG: ', num2str(accuracyNBWithoutHOG)]);
        if accuracyNBWithoutHOG > bestSingleModelWithoutHOGAccuracy
end
            bestSingleModelWithoutHOGAccuracy = accuracyNBWithoutHOG;
            bestSingleModelWithoutHOG = naiveBayesModelWithoutHOG;
            bestSingleModelWithoutHOGName = 'nb without hog';
            knnModelWithoutHOG = fitcknn(XTrain', balancedTrainLabels, 'NumNeighbors', numNeighbors);
            predictionsKNNWithoutHOG = predict(knnModelWithoutHOG, XTest');
            accuracyKNNWithoutHOG = sum(predictionsKNNWithoutHOG == testLabels) / numel(testLabels);
            disp(['kNN Accuracy without HOG with ', num2str(numNeighbors), ' neighbors: ', num2str(accuracyKNNWithoutHOG)]);
            if accuracyKNNWithoutHOG > bestSingleModelWithoutHOGAccuracy
end
                bestSingleModelWithoutHOGAccuracy = accuracyKNNWithoutHOG;
                bestSingleModelWithoutHOG = knnModelWithoutHOG;
                bestSingleModelWithoutHOGName = 'knn without hog';
            if iscell(predictionsRFWithoutHOG)
end
                predictionsRFWithoutHOG = cellfun(@str2double, predictionsRFWithoutHOG);
            predictionsSVMWithoutHOG = double(predictionsSVMWithoutHOG);
            predictionsRFWithoutHOG = double(predictionsRFWithoutHOG);
            predictionsNBWithoutHOG = double(predictionsNBWithoutHOG);
            predictionsKNNWithoutHOG = double(predictionsKNNWithoutHOG);
            combinedPredictionsWithoutHOG = mode(cat(2, double(predictionsSVMWithoutHOG), predictionsRFWithoutHOG, double(predictionsNBWithoutHOG), double(predictionsKNNWithoutHOG)), 2);
            finalPredictionsWithoutHOG = arrayfun(@(x) categorical(x, [1 2], {'normal', 'pneumonia'}), combinedPredictionsWithoutHOG);
            ensembleAccuracyWithoutHOG = sum(finalPredictionsWithoutHOG == testLabels) / numel(testLabels);
            disp(['Ensemble Accuracy without HOG with ' num2str(numTrees) ' trees, ' num2str(numNeighbors), ' neighbors: ', num2str(ensembleAccuracyWithoutHOG)]);
            if ensembleAccuracyWithoutHOG > bestCombinedModelWithoutHOGAccuracy
end
                bestCombinedModelWithoutHOGAccuracy = ensembleAccuracyWithoutHOG;
                bestCombinedModelWithoutHOG = struct('SVM', svmModelWithoutHOG, 'RF', randomForestModelWithoutHOG, 'NB', naiveBayesModelWithoutHOG, 'kNN', knnModelWithoutHOG);
                bestCombinationWithoutHOGDetails.NumTrees = numTrees;
                bestCombinationWithoutHOGDetails.NumNeighbors = numNeighbors;
disp('Best Single Model with HOG:');
disp(['Model: ', bestSingleModelWithHOGName]);
disp(['Accuracy: ', num2str(bestSingleModelWithHOGAccuracy)]);
disp('Best Combined Model with HOG:');
disp(['Model: ', bestCombinedModelWithHOGName]);
disp(['Accuracy: ', num2str(bestCombinedModelWithHOGAccuracy)]);
disp('Best Single Model without HOG:');
disp(['Model: ', bestSingleModelWithoutHOGName]);
disp(['Accuracy: ', num2str(bestSingleModelWithoutHOGAccuracy)]);
disp('Best Combined Model without HOG:');
disp(['Model: ', bestCombinedModelWithoutHOGName]);
disp(['Accuracy: ', num2str(bestCombinedModelWithoutHOGAccuracy)]);
disp('Best Combination with HOG Details:');
disp(['NumTrees: ', num2str(bestCombinationWithHOGDetails.NumTrees)]);
disp(['NumNeighbors: ', num2str(bestCombinationWithHOGDetails.NumNeighbors)]);
disp('Best Combination without HOG Details:');
disp(['NumTrees: ', num2str(bestCombinationWithoutHOGDetails.NumTrees)]);
disp(['NumNeighbors: ', num2str(bestCombinationWithoutHOGDetails.NumNeighbors)]);
saveFolderPath = 'C:\Users\Aravind\Desktop\ROE\trimester 2\Application for data science\Coursework 1\models';
save(fullfile(saveFolderPath, 'bestSingleModelWithHOG.mat'), 'bestSingleModelWithHOG');
save(fullfile(saveFolderPath, 'bestCombinedModelWithHOG.mat'), 'bestCombinedModelWithHOG');
save(fullfile(saveFolderPath, 'bestSingleModelWithoutHOG.mat'), 'bestSingleModelWithoutHOG');
save(fullfile(saveFolderPath, 'bestCombinedModelWithoutHOG.mat'), 'bestCombinedModelWithoutHOG', '-v7.3');
function hogFeatures = extractHOGFeaturesFromImage(image, numBins)
    [featureVector, ~] = extractHOGFeatures(image, 'NumBins', numBins);
    hogFeatures = featureVector;
% Create a subplot for visualization
% Creating subplots for data visualization
% Displaying images
% Add a title to the plot
% Adding titles to plots
% Calculating the accuracy of the model
