function main(path)

path = strcat(path,'\clustering.csv');

data = importdata('clustering.csv');

acc = zeros(20,1);
predictions = zeros(50,200);

% Main loop for different starting points

for k = 1:1000
    [prediction] = KM(data);
    predictions(k,:) = prediction;
    labels = data(:,3);
    errors = abs(labels ~= prediction);
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
hold on 
title('Question 2)A cluster prediction plot')

c1x = sum(x,1)./size(x,1); 
c1y = sum(y,1)./size(y,1); 
c2x = sum(x1,1)./size(x1,1); 
c2y = sum(y1,1)./size(y1,1); 

plot(c1x,c1y, 'b x')
hold on
plot(c2x,c2y, 'b o')

%Confusion matrix

ConfusionM = zeros(2,2);

for i=1:200
   ConfusionM(labels(i), prediction(i)) = ConfusionM(labels(i),prediction(i)) + 1;
end

disp('Confusion matrix is as follows:');
disp(ConfusionM);


% K-means algorithm

function [decision] = KM(cluster)

x1 = rand()*20-10;
y1 = rand()*20-10;
x2 = rand()*20-10;
y2 = rand()*20-10;

center1 = [x1 y1];
center2 = [x2 y2];

for j = 1:200

distance1 = sqrt((cluster(:,1)-center1(:,1)).^2+(cluster(:,2)-center1(:,2)).^2);
distance2 = sqrt((cluster(:,1)-center2(:,1)).^2+(cluster(:,2)-center2(:,2)).^2);

decision = zeros(size(distance1,1),1);
decision(distance1 > distance2) = 1;
decision(distance1 < distance2) = 2;

class1Sum = sum(decision);
class2Sum = 200 - class1Sum;

x = find(decision==1);
class1 = cluster(x,1:2);
class1 = sum(class1)/class1Sum;

xx = find(decision==2);
class2 = cluster(xx,1:2);
class2 = sum(class2)/class2Sum;

end

end

end
