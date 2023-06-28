% Description: This function is used to calculate the cross-entropy loss
% between the predicted y values and the actual y values. The function
% inputs the predicted and the actual values and outputs a vector that
% contains the loss between each column of AL and Y. 

function cost = compute_cost(AL, Y)
    % obtain number of columns of AL : same as columns of Y
    [~,N] = size(AL);
    % initialize cost variable
    cost = zeros(1,N);
    for n = 1:N
        % each entry of cost is the cross-entropy function evaluated at
        % each column of Y and AL. We use .* for element wise
        % multiplication between vectors.
        cost(1,n) = - (sum(Y(:,n).*log(AL(:,n))));
    end
end