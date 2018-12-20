function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];
l1nodes = size(Theta1, 1);
l2nodes = size(Theta2, 1);
for r=1:m, % for each input vector
    inputs = X(r,:)';

    % then, multiple that input vector by the weighting for each layer1 node, 
    % sum it up, apply the sigmoid function, and apply that to the node's cell
    layer1 = zeros(l1nodes, 1);
    for nodeindex=1:l1nodes,
        weightings = Theta1(nodeindex, :);
        cellvalue = weightings * inputs; % cellvalue should be 1x1 or scalar
        layer1(nodeindex, 1) = sigmoid(cellvalue);
    end

    % do it again for layer2
    layer2 = zeros(l2nodes, 1);
    % prepend a bias entry to layer1
    layer1 = [1; layer1];
    for nodeindex=1:l2nodes,
        weightings = Theta2(nodeindex, :);

        cellvalue = weightings * layer1; % cellvalue should be 1x1 or scalar
        layer2(nodeindex, 1) = sigmoid(cellvalue);
    end

    % layer2 contains the predicted probability that each class is the
    % correct one. Find the most likely and use that as the prediction.
    [dummy prow] = max(layer2);
    p(r) = prow;
end








% =========================================================================


end
