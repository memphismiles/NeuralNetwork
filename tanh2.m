% Description: This function is the activation function in the neural
% networks for all steps except the last one. The function inputs a matrix
% X and applies the hyperbolic tangent function to each element of X.

function Z = tanh2(X)
    % obtain size of X
    [M, N] = size(X);
    % initialize Z array as zeros with same size as X
    Z = zeros(M,N); 
    for i = 1:M
        for j = 1:N
            % update the (i,j) entry of Z with the tanh function applied on
            % X(i,j). We use a slightly different formula to avoid it
            % blowing up.
            Z(i,j) = 2/(1+exp(-2*X(i,j))) - 1;
        end
    end
end
% function returns a Z with every entry representing the same entry in X
% with the function applied. There probably is a more efficient way to do
% this, but this makes sense in my head without using some sort of function
% mapping or anonymous function declaring.