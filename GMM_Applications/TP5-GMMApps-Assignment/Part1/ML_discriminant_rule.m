function [Y_est] = ML_discriminant_rule(X, classes, models, K)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[~, M] = size(X);
numClasses = length(classes);
Y_est= zeros(1,M);

for m=1:M
    likelihood = zeros(numClasses,1);
    for i=1:numClasses
        for k=1:K
            pdf = my_gaussPDF(X(:,m), models(i).Mu(:,k), models(i).Sigma(:,:,k));
            likelihood(i) = likelihood(i) + pdf;
        end
        likelihood(i) = -1*log(likelihood(i));
    end

    Y_est(m) = classes(likelihood == min(likelihood));
end

end

