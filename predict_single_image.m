% Description: This function is provided to us in the assignment. It is
% used to display what the model specifically does. It picks a random
% image, and shows the image alongside a bar chart of what our model
% predicts the image to be.


function predict_single_image(parameters)
    % load test images
    load('test_images.mat')
    % pick random image
    index = randi(length(pixel));
    % reshape and normalize the image
    X = reshape(pixel(:,:,index), [size(pixel,1)*size(pixel,2), 1]) / 255;
    % run image through model
    forward = forward_propagation(X, parameters);
    % obtain probabilities
    probabilities = forward{end};

    % plot probabilities in bar plot and show image
    figure();
    subplot(1,2,1);
    imshow(pixel(:, :, index))
    title('Input Image')
    subplot(1,2,2);
    bar(0:9, probabilities)
    xlabel('Classes');
    ylabel('Probability');
    title('Probability Distribution')

    % Used to save for using in report
    saveas(h1, 'predict_single_image.png')
end