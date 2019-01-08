% *****  QUESTION 3-1 Linear SVM ***********************
function main(path)

path = strcat(path,'\HW2data.mat')
load(path)

% Shuffling indices of P and B in the first dimension
randP = Ps(randperm(size(Ps,1)),:);
randB = Bs(randperm(size(Bs,1)),:);

trainSizeP = floor(size(Ps,1)*0.7);
trainSizeB = floor(size(Bs,1)*0.7);
testSetSize = (size(Ps,1) + size(Bs,1) - trainSizeP - trainSizeB);

trainSet = zeros(trainSizeP + trainSizeB,16);
testSet = zeros(testSetSize,16);

% Putting P and B train points to the train set according to these random
% indices
trainSet(1:trainSizeP,:) = randP(1:trainSizeP,:);
trainSet(trainSizeP + 1:end,:) = randB(1:trainSizeB,:);

% Putting P and B test points to the test set according to these random
% indices
testSet(1:size(randP,1) - trainSizeP,:) = randP(trainSizeP + 1:end,:);
testSet(size(randP,1) - trainSizeP + 1:end,:) = randB(trainSizeB + 1:end,:);

%Creating the appropiate labels for each train and test set data point
trainLabels = zeros(trainSizeP + trainSizeB,1);
trainLabels(trainSizeP + 1:end) = 1;

testLabels = zeros(size(testSet,1),1);
testLabels(size(randP,1) - trainSizeP + 1:end) = 1;

save('testSet.mat','testSet')
save('trainSet.mat','trainSet')
save('testLabels.mat','testLabels')
save('trainLabels.mat','trainLabels')


% C values for the linear SVM
C=[10^-4 10^-3 10^-2 10^-1 1 10 100];

% This matrix will include the accuracy values for different C values
Accuracy=zeros(1,7);

for i = 1:7
    
    svm_Linear = fitcsvm(trainSet,trainLabels,'BoxConstraint',C(i));
    accuracyCV = crossval(svm_Linear); % Cross validation accuracy
    
    classLoss = kfoldLoss(accuracyCV); % 
    Accuracy(i) = 1 - classLoss;
    
end

figure()
plot(Accuracy)
title('Question 3-1) Accuracy of Cross Validation vs Chosen C value')
ylabel('Accuracy of Cross Validation')
xlabel(' Index of the C value used (refer to the C array in the code))')

maxAccC = find(Accuracy == max(Accuracy(:))); % C that gives the maximum accuracy;
disp('C value with the highest accuracy C= ')
disp(maxAccC)

svm_Linear = fitcsvm(trainSet,trainLabels,'BoxConstraint',maxAccC);
linearPrediction = predict(svm_Linear,testSet);

testAccuracy = (testLabels - linearPrediction)== 0;
testAccuracy = sum(testAccuracy)/size(testLabels,1);

disp('Test accuracy for the best C value: ');
disp(testAccuracy);

% Outputting the results
disp('Prediction results are saved to linear_SVM_Results.csv file');
csvwrite('linear_SVM_Results.csv',linearPrediction);

end
