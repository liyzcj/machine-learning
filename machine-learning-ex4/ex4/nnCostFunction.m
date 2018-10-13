function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Part 1 : forward propagation===>  m is the number of train data
A1 = X; % dim = (m * 400)
A1 = [ones(m, 1) A1]; % dim = (m * 401) 
H2 = Theta1 * A1'; % H2.dim = (25 * m) | Theta1.dim = (25 * 401) | A1'.dim = (401 * m)
A2 = sigmoid(H2); % A2.dim = (25 * m)
A2 = [ones(1, size(A2, 2)); A2]; % A2.dim = (26 * m)
H3 = Theta2 * A2; % H3.dim = (10 * m) | Theta2.dim = (10 * 26) | A2.dim = (26 * m)
A3 = sigmoid(H3); % A3.dim = (10 * m)
Y = repmat(1:num_labels,m,1); % Y.dim = (m * 10)
Y = Y == y; % Y.dim = (m * 10)
% Vectorized way to comput cost J
J = - Y' .* log(A3) - (1 - Y)' .* log(1 - A3);
J = sum(J(:)) / m;
% Not vectorized way
% for i = 1:5000
%     for j = 1:10
%         J = J +( -Y(i,j) * log(A3(j,i)) - (1 - Y(i,j)) * log(1 - A3(j,i)));
%     end
% end
% J = J / m;

% Cost Function with Regularized
Reg1 = Theta1(:,2:end) .^ 2;
Reg2 = Theta2(:,2:end) .^ 2;
Reg = (sum(Reg1(:)) + sum(Reg2(:))) * lambda / (2 * m);
J = J + Reg;

% Part 2 : back propagation == > 

delta3 = A3 - Y';
Temp  = Theta2' * delta3;
Temp = Temp(2:end,:);
delta2 = Temp .* sigmoidGradient(H2);
Delta1 = delta2 * A1;
Delta2 = delta3 * A2';
Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

% Part 3 : Regularized Gradient
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end) * lambda / m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end) * lambda / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
