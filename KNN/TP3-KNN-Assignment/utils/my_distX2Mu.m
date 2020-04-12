function [d] =  my_distX2Mu(X, Mu, params)
%MY_DISTX2Mu Computes the distance between X and Mu.
%
%   input -----------------------------------------------------------------
%   
%       o X     : (D x N), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Mu    : (D x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N
%       o params : struct array containing the parameters of the KNN (k,
%                  d_type and eventually the parameters for the Gower
%                  similarity measure)
%
%   output ----------------------------------------------------------------
%
%       o d      : (k x N), distances between X and Mu 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
[D, N] = size(X);
[~, k] = size(Mu);

% Output Variable
d = zeros(k,N);

% Output Variable
for i=1:k
    for j = 1:N
        d(i,j) = my_distance(X(:,j), Mu(:,i), params.d_type);        
    end
end


end