% Description: This function is used to load in the datasets X_train,
% Y_train, X_test, and Y_test. The original files provided in the
% assignment have the data stored in cells. This function is used to access
% those matrices inside the cells and assign them accordingly. This
% function also reshapes and normalizes the image data in X_train and X_test. 
% Lastly, it performs one-hot encoding on the labels.

function [X_train, Y_train, X_test, Y_test] = load_train_and_test_data()
    load('train_images.mat'); % load in train images
    [L, W, N] = size(pixel); % obtain sizes
    X_train = zeros(L*W, N); % initialize X_train
    for i = 1:N 
        image = reshape(pixel(:,:,i), [L*W,1]); % reshaping the i-th image
        image = normalize(image); % normalize the i-th image
        X_train(:,i) = image; % store the normalized, reshaped data into the i-th column
    end
    load('train_labels.mat'); % load in train labels
    Y_train = zeros(10, N); % initialize Y_train
    for j = 1:N
        num = label(j); % obtain label of the j-th image
        Y_train(num+1,j) = 1; % one-hot encoding
    end

    % repeat process for x_test and y_test

    load('test_images.mat'); % load in test images
    [L2, W2, N2] = size(pixel); % obtain new sizes
    X_test = zeros(L2*W2, N2); % initialize X_test
    for i = 1:N2 
        image = pixel(:,:,i); % storing the i-th image
        image = reshape(image, [L2*W2,1]); % reshaping the i-th image
        image = normalize(image); % normalize the i-th image
        X_test(:,i) = image; % store the normalized, reshaped data into the i-th column
    end
    load('test_labels.mat'); % load in train labels
    Y_test = zeros(10, N2); % initialize Y_test
    for j = 1:N2
        num = label(j); % obtain label of the j-th image
        Y_test(num+1,j) = 1; % one-hot encoding
    end
end
