function [XNew] = sample_per_class(models, class, nbPoints)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Kparam = size(models(class).Priors,2);
XNew = [];
for j=1:(nbPoints)
        k = randsrc(1,1,[1:Kparam; models(class).Priors]);
        XNew = [XNew, mvnrnd(models(class).Mu(:,k)', models(class).Sigma(:,:,k))'];
end
end

