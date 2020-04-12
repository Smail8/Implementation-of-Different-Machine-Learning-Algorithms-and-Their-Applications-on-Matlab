function [Priors,Mu,Sigma] = maximization_step(X, Pk_x, params)
%MAXIMISATION_STEP Compute the maximization step of the EM algorithm
%   input------------------------------------------------------------------
%       o X         : (N x M), a data set with M samples each being of 
%       o Pk_x      : (K, M) a KxM matrix containing the posterior probabilty
%                     that a k Gaussian is responsible for generating a point
%                     m in the dataset, output of the expectation step
%       o params    : The hyperparameters structure that contains k, the number of Gaussians
%                     and cov_type the coviariance type
%   output ----------------------------------------------------------------
%       o Priors    : (1 x K), the set of updated priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu        : (N x K), an NxK matrix corresponding to the updated centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
%                   updated Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%

% Additional variables
[N, M] = size(X);
Priors = zeros(1,params.k);
Mu = zeros(N,params.k);
Sigma = zeros(N,N,params.k);
eps = 1e-5;
K = params.k;
sum_Pk_x = sum(Pk_x');
Priors = sum_Pk_x/M;
Mu = (X*(Pk_x'))./sum_Pk_x;

for k=1:K
    sum3 = 0;
    if params.cov_type == "full"
        for i=1:M
            sum3 = sum3 + (Pk_x(k,i)*(X(:,i)-Mu(:,k))*transpose(X(:,i)-Mu(:,k)));
        end
        Sigma(:,:,k) = sum3/sum_Pk_x(k);
    elseif params.cov_type == "diag"
        for i=1:M
            sum3 = sum3 + (Pk_x(k,i)*(X(:,i)-Mu(:,k))*transpose(X(:,i)-Mu(:,k)));
        end
        Sigma(:,:,k) = diag(diag(sum3/sum_Pk_x(k)));
    elseif params.cov_type == "iso"
         for i=1:M
            sum3 = sum3 + (Pk_x(k,i)*(norm(X(:,i)-Mu(:,k))^2));
         end
        Sigma(:,:,k) = (sum3/(N*sum_Pk_x(k)))*eye(N);
    end

    Sigma(:,:,k) = Sigma(:,:,k) + eps*eye(N); 

    
    
end

end

