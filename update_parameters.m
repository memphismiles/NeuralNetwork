% Description: This function uses the gradient descent updating step on the
% parameters of our model. Using a learning rate and the gradient of the
% loss function, we update each parameter to move in the direction of the
% minimum of the loss function. An updated parameters cell is returned.

function parameters = update_parameters(parameters,gradients,learning_rate)
    % obtain length of parameters
    L = length(parameters);
    for i = 1:L
        % Update each weights and biases matrices by using gradient
        % descent. We use for loop to get through all elements of
        % parameters efficiently.
        parameters{i}.W = parameters{i}.W - learning_rate*gradients{i}.dW;
        parameters{i}.b = parameters{i}.b - learning_rate*gradients{i}.db;
    end
end