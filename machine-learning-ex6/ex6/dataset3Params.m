function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_matrix = eye(64, 3);
i = 0;

for C = list
    for sigma = list
        i = i + 1;
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        
        % filling the error matrix :
        error_matrix(i,:) = [C, sigma, prediction_error];
    end
end

results = sortrows(error_matrix, 3);  % sort the rows of the matrix in ascending
                                    % order based on the values of the 3rd
                                    % column. It gives a matrix containing
                                    % the C and sigma which have the 
                                    % smallest error in the first row of
                                    % this so called matrix.
C = results(1,1);
sigma = results(1,2);

% =========================================================================

end
