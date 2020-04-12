function [models] = learn_gmm(X, Y, params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
classes = unique(Y);
numClasses = length(classes);

for i=1:numClasses
    [models(i).Priors, models(i).Mu, models(i).Sigma, ~, ~, ~, ~] = my_gmmEM(X(:, Y == classes(i)), params);
end

end

