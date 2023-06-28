% Description: This function is the last activation function in the neural
% network. It maps a matrix X into a probability vector, where each element
% in the vector adds up to 1.

function Z = softmax(X)
    % obtain size of X
    [K,N] = size(X);
    % initialize Z with zeros in same size as X
    Z = zeros(K,N);
    for n = 1:N
        for k = 1:K
            % for the (n,k) entry, we calculate e^x and divide it by the
            % sum across the column. This maps X to a probability distribution 
            % over the classes 
            Z(k,n) = exp(X(k,n)) / sum(exp(X(:,n)));
        end
    end
end