function [XNew] = draw_samples(models,nbPoints)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Kparam = size(models(1).Priors,2);
numClasses = size(models,2);
display(numClasses);
XNew = [];
for i=1:numClasses
    for j=1:(nbPoints/numClasses)
        k = randsrc(1,1,[1:Kparam; models(i).Priors]);
        XNew = [XNew, mvnrnd(models(i).Mu(:,k)', models(i).Sigma(:,:,k))'];
    end
end

end

