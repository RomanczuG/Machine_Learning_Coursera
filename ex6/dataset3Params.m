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

indexRow = 1;
X1 = X(:,1);
X2 = X(:,2);
results = ones(64, 3);

for C_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
     
% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.
    for sigma_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
        model= svmTrain(X, y, C_test, @(X1, X2) gaussianKernel(X1, X2, sigma_test));
        predictions = svmPredict(model,Xval);
        error = mean(double(predictions ~= yval));
        results(indexRow, :) = [error, C_test, sigma_test];
        indexRow = indexRow + 1;
        
    end
end
results;
sortedResults = sortrows(results, 1);

C = sortedResults(1,2);
sigma = sortedResults(1,3);

% =========================================================================

end
