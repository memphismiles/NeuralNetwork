% Description: This function is used to initialize the parameters for the
% neural network. It uses the dimensions of each layer to create the
% weights and biases of each layer, using random standard uniform values
% for weights and zeros for biases. All the parameters are stored in a
% cell, where each struct holds the weights and biases of each layer.

function parameters = initialize_parameters(layer_dims)
    % completed this using code in lecture
    % First element of parameters is a struct with field W and b, to
    % represent the weights and biases of the transition from first layer to second.
    % The W has dimension: 2nd element of layer_dims x 1st element of layer_dims. 
    % This is because W is used to take one layer to the next one, i.e. 
    % from R^(1st element) to R^(2nd element).
    % b is a one-dimensional array with length of 2nd element of layer_dims

    % use loop to accomplish this for all elements of cell
    L = length(layer_dims);
    % initialize parameters cell
    parameters = cell(L-1,1);

    for l = 1:L-1
        % create matrix with random numbers from 0-1
        parameters{l}.W = randn(layer_dims(l+1),layer_dims(l)); 
        % create vector with zeros for b
        parameters{l}.b = zeros(layer_dims(l+1), 1);
    end
end