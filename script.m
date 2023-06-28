%
%
%
% Description: This is the main script for training and testing the model.
% After loading in the data and definining hyperparameters, we can train
% the model. First, we initialize parameters, the weights and biases of
% the neural network. Next, we use mini-batch gradient descent to train the
% model 150 times. To do this, we call the functions,
% forward_propagation(), backward_propagation(), and update_parameters().
% We use compute_cost() to calculate cost and store all the training
% losses. We also store the accuracies using predict() and accuracy().
% After training the data, we can call visualize_history() to see the
% training loss vs. epochs and testing accuracy vs. epochs.
%

clc; close all; clear all;

% Loading in Training and Testing Data
[X_train, Y_train, X_test, Y_test] = load_train_and_test_data();

% Defining hyperparameters
input_size = size(X_train,1);
output_size = size(Y_train,1);
neurons = 64;
numLayer = 2;
lr = 0.01;
epochs = 150;

% Define layer_dims
layer_dims = zeros(1, numLayer + 2);
layer_dims(1) = input_size;
layer_dims(end) = output_size;
for i = 1:numLayer
    layer_dims(i+1) = neurons;
end

% Training the model
parameters = initialize_parameters(layer_dims);

% Use mini-batch GD
m = size(X_train, 2);
batch_size = 64;
num_batches = floor(m / batch_size);
% initialize trainLoss and testAccuracy for plotting later
trainLoss = zeros(epochs,1);
testAccuracy = zeros(epochs,1);

for i = 1:epochs
    % randomize X_train and Y_train order
    indices = randperm(m);
    X_train = X_train(:, indices);
    Y_train = Y_train(:, indices);
    
    % initialize cost
    cost = zeros(num_batches, batch_size);
    
    % go through each batch until all observations are seen by model
    for j = 1:num_batches
        X_batch = X_train(:, (j-1)*batch_size+1:j*batch_size);
        Y_batch = Y_train(:, (j-1)*batch_size+1:j*batch_size);
        forward = forward_propagation(X_batch, parameters);
        cost(j,:) = compute_cost(forward{end}, Y_batch);
        gradients = backward_propagation(X_batch, Y_batch, parameters, forward);
        parameters = update_parameters(parameters, gradients, lr);
    end
    % calculate accuracy
    Y_pred=  predict(X_test, parameters);
    acc = accuracy(Y_pred, Y_test);

    fprintf('Loss after epoch %d: Training: %f\n', i, norm(cost));
    % update i-th entry of trainLoss and testAccuracy
    trainLoss(i) = norm(cost);
    testAccuracy(i) = acc;
end

% print ending testAccuracy
fprintf('Test accuracy: %f\n', testAccuracy(end));

% create plots of train loss and accuracy vs epochs
visualize_history(epochs, trainLoss, testAccuracy, lr, numLayer);