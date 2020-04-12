function [y_est, var_est] = my_gmr(Priors, Mu, Sigma, X, in, out)
%MY_GMR This function performs Gaussian Mixture Regression (GMR), using the 
% parameters of a Gaussian Mixture Model (GMM) for a D-dimensional dataset,
% for D= N+P, where N is the dimensionality of the inputs and P the 
% dimensionality of the outputs.
%
% Inputs -----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM 
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the 
%              K GMM components.
%   o X:       N x M array representing M datapoints of N dimensions.
%   o in:      1 x N array representing the dimensions of the GMM parameters
%                to consider as inputs.
%   o out:     1 x P array representing the dimensions of the GMM parameters
%                to consider as outputs. 
% Outputs ----------------------------------------------------------------
%   o y_est:     P x M array representing the retrieved M datapoints of 
%                P dimensions, i.e. expected means.
%   o var_est:   P x P x M array representing the M expected covariance 
%                matrices retrieved. 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N,M] = size(X);
[~,P] = size(out);
[~,K] = size(Priors);
D = N+P;
y_est = zeros(P,M);
var_est = zeros(P,P,M);

for m=1:M
    for k=1:K
        sum = 0;
        for i=1:K
           sum = sum + (Priors(1,i) * my_gaussPDF(X(:,m), Mu(1:N,i), Sigma(1:N,1:N,i)));
        end

        beta = (Priors(1,k)*my_gaussPDF(X(:,m), Mu(1:N,k), Sigma(1:N,1:N,k)))/sum;
        Mu_tilt = Mu(N+1:D,k) + (Sigma(N+1:D,1:N,k)*inv(Sigma(1:N,1:N,k))*(X(:,m)-Mu(1:N,k)));
        Sigma_tilt = Sigma(N+1:D,N+1:D,k) - (Sigma(N+1:D,1:N,k)*inv(Sigma(1:N,1:N,k))*Sigma(1:N,N+1:D,k));
        y_est(:,m) = y_est(:,m) + (beta*Mu_tilt);
        var_est(:,:,m) = var_est(:,:,m) + (beta*(Mu_tilt.^2 + Sigma_tilt));
    end
    var_est(:,:,m) = var_est(:,:,m) - (y_est(:,m).^2);
end















end