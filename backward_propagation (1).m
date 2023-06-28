% Description: This function is used to find the gradient of the loss
% function with respect to each of the parameters. These are found using
% chain rule, and the math is laid out in the report. The function outputs
% a cell called gradients, and each struct of gradients contains the
% partial derivatives at each layer. The gradients depend on X, Y,
% parameters, and the output of forward_propagation(), so those are the
% inputs.

function gradients = backward_propagation(X,Y, parameters, activations)
    % obtain length of parameters (number of layers)
    L = length(parameters);
    % obtain number of columns in X
    m = size(X,2);
    % initialize the cell gradients
    gradients = cell(1,L);

    % using formulas in lecture and in my report, we find that it's easiest
    % to find the gradient of the last set of weights and biases first
    dZ = activations{end} - Y;
    gradients{L}.dW = dZ * activations{L}'/m;
    gradients{L}.db = sum(dZ,2)/m;

    % Formulas for gradients all depend on the gradient in front. Thus, we
    % loop from (L-1) down to 1
    for l = (L-1):-1:1
        % Use formulas
        dA = parameters{l+1}.W' * dZ;
        dZ = dA .* (1- tanh2(activations{l+1}).^2);
        if l == 1
            % need an A_prev for l=1, which would be input vector X
            A_prev = X;
        else
            A_prev = activations{l};
        end
        gradients{l}.dW = dZ * A_prev'/m;
        gradients{l}.db = sum(dZ,2)/m;
    end
end