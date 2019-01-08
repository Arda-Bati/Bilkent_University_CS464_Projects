% *****  QUESTION 3-2 Gaussian SVM ***********************
function main(path)

clear variables

load('testLabels.mat')
load('trainLabels.mat')
load('testSet.mat')
load('trainSet.mat')

%SVM with Gaussian kernel

gamma = [2^-4 2^-3 2^-2 2^-1 2^0];
kernelScale = 1./(gamma); %Kernel Scale parmeter used in fitcsvm below 
%divides the kernel function's argument by the scale specified. scale is 
%set to 1/gamma to multiply the function with gamma, as required

C=[10^-3 10^-2 10^-1 1 10];
 
Accuracy = zeros(5,5);

for i = 1:5
    
    for j = 1:5
        
        svm_Gaussian = fitcsvm(trainSet,trainLabels,'BoxConstraint',C(i),'KernelFunction', 'gaussian','KernelScale',kernelScale(j));
        accuracyCV = crossval(svm_Gaussian);
        
        classLoss = kfoldLoss(accuracyCV); % 
        Accuracy(i,j) = 1 - classLoss;
    end
    
end

% Drawing Surface Plot
figure()
surf(gamma,C,Accuracy)
xlabel('gamma')
ylabel('C')
zlabel('Accuracy of Cross Validation')
title('Question 3-2) Accuracy of cross validation vs Gamma vs C Value')

[i1,i2] = find(Accuracy == max(Accuracy(:))); % Best C value - gamma pair

disp('Best C value: ')
disp(i1)
disp('Best gamma value: ')
disp(i2)
disp('Accuracy of the given C and gamma pair')
disp(max(Accuracy(:)))

svm_Gaussian = fitcsvm(trainSet,trainLabels,'BoxConstraint',C(i),'KernelFunction', 'gaussian','KernelScale',kernelScale(j));
gaussianPrediction = predict(svm_Gaussian,testSet);
%finding test accuracy

testAccuracy = (testLabels - gaussianPrediction)== 0;
testAccuracy = sum(testAccuracy)/size(testLabels,1);

disp('Test set accuracy: ')
disp(testAccuracy)

% Outputting the results
disp('Prediction results are saved to gaussian_SVM_Results.csv file');
csvwrite('gaussian_SVM_Results.csv',gaussianPrediction);

end
