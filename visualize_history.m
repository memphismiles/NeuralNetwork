% Description: This function is called at the end of the script to
% visualize the convergence of our model. Two plots are shown: trainLoss vs
% epochs and testAccuracy vs epochs. The title is customized to the
% parameters provided. The image is also saved as a customized name.

function visualize_history(epochs, trainLoss, testAccuracy, learning_rate, numLayer)
    % create figure   
    h1 = figure();
    % plot first graph on the left
    subplot(1,2,1);
    plot(1:epochs, trainLoss)
    xlabel('Epochs')
    ylabel('Training Loss')
    xticks([0,50,100,150,200,250,300])
    % plot second graph on the right
    subplot(1,2,2);
    plot(1:epochs, testAccuracy)
    xlabel('Epochs')
    ylabel('Test Accuracy')
    xticks([0,50,100,150,200,250,300])
    pos = get(gca, 'Position');
    pos(1) = pos(1) + 0.05;
    set(gca, 'Position', pos)
    
    % create title of entire subplot
    titlestring = sprintf('Epochs: %g; learning rate: %.2f; number of hidden layers: %g', epochs, learning_rate, numLayer);
    sgtitle(titlestring, 'FontSize', 11);


    % save as with informative name
    savestring = sprintf('model_%.2f_%g_%g.png', learning_rate, numLayer, epochs);
    saveas(h1, savestring);

end