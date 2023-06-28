% Description: This function is used to run through the neural network. It
% takes an input matrix X through the neural network. At each layer, the
% vector is linearly transformed with the corresponding struct of
% parameters and put through an activation function. At the end of the
% neural network, the input matrix X is transformed into a matrix of
% probabilities. All layers are stored inside a cell called activations,
% which is returned by this function.

function activations = forward_propagation(X, parameters)
    % obtain length of parameters (number of layers)
    L = length(parameters);
    % initialize our A matrix as our input
    A = X;
    % create activations cell
    activations = cell(1, L+1);
    % initialize first struct in activations with the input layer
    activations{1} = X;
    % loop through each layer
    for l = 1:L
        % Use formula A = activation function applied to WX + b
        Z = parameters{l}.W * A + parameters{l}.b;
        % If last layer, then the activation function is softmax
        if l == L
            A = softmax(Z);
        % if not last layer, the activation function is tanh2
        else
            A = tanh2(Z);
        end
        % set l+1 entry of activations struct as A (the next layer)
        activations{l+1} = A;
    end
end