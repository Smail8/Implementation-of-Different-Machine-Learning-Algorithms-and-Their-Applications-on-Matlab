function [ logl ] = my_gmmLogLik(X, Priors, Mu, Sigma)
%MY_GMMLOGLIK Compute the likelihood of a set of parameters for a GMM
%given a dataset X
%
%   input------------------------------------------------------------------
%
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                    Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%
%   output ----------------------------------------------------------------
%
%      o logl       : (1 x 1) , loglikelihood
%%


% Auxiliary Variables
[N, M] = size(X);
[~, K] = size(Mu);

logl = 0;

for i=1:M
    sum = 0;
   for k=1:K
      sum = sum + (Priors(1,k)*my_gaussPDF(X(:,i), Mu(:,k), Sigma(:,:,k)));
   end
   logl = logl + log(sum);
end


end

