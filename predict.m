% Description: This function is used to find what our model predicts an
% image to be. It runs a matrix X through the model and outputs the matrix
% of our model's prediction. Each column contains the probability vector
% that the neural network outputs.

function Y_pred = predict(X, parameters)
    % obtain number of columns of X
    N = size(X, 2);
    % run forward propagation with finalized parameters. This will output
    % the layers of X through the neural network
    forward = forward_propagation(X, parameters);
    % The last struct of forward holds the output probability vector of X
    % through the neural network
    probabilities = forward{end};
    % initialize Y_pred as a 10XN matrix of zeros. We will change one entry
    % per column to be a 1.
    Y_pred = zeros(10, N);
    for n = 1:N
        % The highest probability is the value that the model predicts
        [~, index] = max(probabilities(:,n));
        % Update the i-th entry as a 1
        Y_pred(index,n) = 1;
    end
end