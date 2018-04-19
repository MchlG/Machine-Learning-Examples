
%polynomial regression with gradient_descent
%polynomial function = theta1*x1 + theta2*x2 + theta3*square_root(x3)

format long %display long form numbers

function norm_x = normalizeFeatures(X)
	temp_x = (X-mean(X))./std(X); 
	temp_x(isnan(temp_x)) = 1; %std(X) caused first column (which is all 1's) to equal NaN, thus we replace all NaN with 1
	norm_x = temp_x;
end

function J = computeCost(X, y, theta)
	m = length(y);
	square_errors = (X*theta-y).^2; % square errors of predicted values
	J = (1/2*m)*sum(square_errors); %value of cost function for given theta
end

function [j_his, theta] = gradientDescent(X, y, theta, alpha, iters)
	m = length(y);
	j_his = zeros(iters, 1);
	for i=1:iters
		theta -= (alpha/m)*(X'*(X*theta-y)); %adjust theta on each iteration by subtracting 1/2m*deriviative_of_cost_function 
		j_his(i, 1) = computeCost(X, y, theta); %save cost values so we can create a plot to visualize value of cost function over each iteration
	end
	j_his = j_his;
	theta = theta;
end


data = load('square-root-data.txt');
figure(1)
plot(data(:, 1), data(:, 2), 'rx', 'markersize', 3, 'markeredgecolor', 'red');
hold on

X = data(:, 1); %get inputs
y = data(:, 2); %get training data outputs
X = [X, X]; %copy inputs as our second feature

X = [ones(size(X, 1), 1), X]; %add column of 1's to be the co-efficient of theta0
X = [X(:, 1), X(:, 2), sqrt(X(:, 3))]; %our second feature is the sqrt(x)
X = normalizeFeatures(X); %normalize the data

theta = [0; 0; 0]; 
alpha = 0.01; %learning rate
iters = 4000; 


[j_his, theta] = gradientDescent(X, y, theta, alpha, iters);

predictions = X*theta; %theta has been adjusted by gradient descent, now create our predicitons from our trained model
plot(data(:, 1), predictions, 'rx', 'markersize', 3, 'markeredgecolor', 'blue');

figure(2) 
plot([1:iters], j_his, 'rx', 'markersize', 1, 'markeredgecolor', 'green'); %plot the cost function. it should decrease over time ie. for ALL convex functions


