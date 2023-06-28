% Description: This function is used to calculate the accuracy of our model
% on a test dataset. It inputs two KxN matrices and calculates how many
% columns out of the N are correct. This count divided by the total number
% of columns is the accuracy of the model.


function acc = accuracy(Y_pred, Y)
    % obtain number of columns : number of observations
    total = size(Y,2);
    % initialize count variable
    count = 0;
    for n = 1:total
        % we need to check if the column of Y_pred is equal to the column
        % of Y. We use the isequal() function to compare vectors
        if isequal(Y_pred(:,n), Y(:,n))
            % increment count if they are equal
            count = count +1;
        end
    end
    % return the probability, which is the number of correct predictions
    % over the total number of observations
    acc = count / total;
end