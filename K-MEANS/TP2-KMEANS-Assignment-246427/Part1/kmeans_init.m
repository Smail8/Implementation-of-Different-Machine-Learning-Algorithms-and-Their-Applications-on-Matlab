function [Mu] =  kmeans_init(X, k, init)
%KMEANS_INIT This function computes the initial values of the centroids
%   for k-means algorithm, depending on the chosen method.
%
%   input -----------------------------------------------------------------
%   
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o k     : (double), chosen k clusters
%       o init  : (string), type of initialization {'random','uniform'}
%
%   output ----------------------------------------------------------------
%
%       o Mu    : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N                   
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
[N, M] = size(X);

% Output Variable
Mu = zeros(N, k);
R = max(X, 2);

if(init == "random")
    Mu = datasample(X, k, 2, 'replace', false);
elseif(init == "uniform")
    for i=1:N
        interval = [min(X(i,:)):0.01:max(X(i,:))];
        Mu(i, :) = randsample(interval, k, false);
    end
end

end