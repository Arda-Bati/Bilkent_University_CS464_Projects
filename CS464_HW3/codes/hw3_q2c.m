function main(path)

path = strcat(path,'\clustering.csv');
data = load(path);

functionRepeatCount = 15;
acc = zeros(functionRepeatCount,1);
predictions = zeros(functionRepeatCount,200);

% Main loop for different starting points

for k = 1:functionRepeatCount
    [prediction] = KM(data);
    predictions(k,:) = prediction;
    labels = data(:,3);
    errors = (labels ~= prediction);
    errors = sum(errors);
    acc(k,1) = 1 - errors/200;
    [m index] = max(acc);
    bestAccDecisions = predictions(index,:);
    
end

disp(' ');
disp(' ');
disp(' ');
disp('Accuracy is:');
disp(m);

% Drawing the cluster plot with best accuracy

figure()
prediction = bestAccDecisions;
x = data(prediction==1,1);
y = data(prediction==1,2);
plot(x,y,'g o');
hold on
x1 = data(prediction==2,1);
y1 = data(prediction==2,2);
plot(x1,y1,'r o');

x1 = data(prediction == 1, 1);
y1 = data(prediction == 1, 1);
x2 = data(prediction == 2, 2);
y2 = data(prediction == 2, 2);

c1x = sum(x1,1)./size(x1,1); 
c1y = sum(y1,1)./size(y1,1); 
c2x = sum(x2,1)./size(x2,1); 
c2y = sum(y2,1)./size(y2,1); 

plot(c1x,c1y, 'b x')
hold on
plot(c2x,c2y, 'b o')

title('Question 2)B cluster prediction plot')

%Confusion matrix

ConfusionM = zeros(2,2);

for i=1:200
   ConfusionM(labels(i), prediction(i)) = ConfusionM(labels(i),prediction(i)) + 1;
end

disp('Confusion matrix is as follows:');
disp(ConfusionM);


function [decision] = KM(cluster)

x1 = (rand()*20-10).^2;
y1 = sqrt(2)*((rand()*20-10).*(rand()*20-10));
z1 = (rand()*20-10).^2;

x2 = (rand()*20-10).^2;
y2 = sqrt(2)*((rand()*20-10).*(rand()*20-10));
z2 = (rand()*20-10).^2;

center1 = [x1 y1 z1];
center2 = [x2 y2 z2];

clusterKernel = zeros(200,4);
clusterKernel(:,1) = cluster(:,1).^2;
clusterKernel(:,2) = sqrt(2)*(cluster(:,1).*cluster(:,2));
clusterKernel(:,3) = cluster(:,2).^2;
clusterKernel(:,4) = cluster(:,3);

for j = 1:2
    
     distance1 = sqrt((clusterKernel(:,1)-center1(1)).^2 + (clusterKernel(:,2)-center1(2)).^2 + (clusterKernel(:,3)-center1(3)).^2);
     distance2 = sqrt((clusterKernel(:,1)-center2(1)).^2 + (clusterKernel(:,2)-center2(2)).^2 + (clusterKernel(:,3)-center2(3)).^2);

    decision = zeros(size(distance1,1),1);
    decision(distance1 > distance2) = 1;
    decision(distance1 < distance2) = 2;

    class1Sum = sum(decision);
    class2Sum = 200 - class1Sum;

    x = find(decision);
    class1 = clusterKernel(x,1:3);
    class1 = sum(class1)/class1Sum;

    xx = find(decision==0);
    class2 = clusterKernel(xx,1:3);
    class2 = sum(class2)/class2Sum;

    center1 = class1;
    center2 = class2;

    end

end

end
