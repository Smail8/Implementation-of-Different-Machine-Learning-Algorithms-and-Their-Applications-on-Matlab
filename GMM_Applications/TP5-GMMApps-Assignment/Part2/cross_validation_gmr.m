function [ MSE_F_fold, NMSE_F_fold, R2_F_fold, AIC_F_fold, BIC_F_fold, std_MSE_F_fold, ...,
    std_NMSE_F_fold, std_R2_F_fold, std_AIC_F_fold, std_BIC_F_fold] = cross_validation_gmr( X, y, ...,
    F_fold, valid_ratio, k_range, params )
%CROSS_VALIDATION_REGRESSION Implementation of F-fold cross-validation for regression algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (P x M) array representing the y vector assigned to
%                           each datapoints
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o valid_ratio  : (double), Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%       o params    : parameter strcuture of the GMM
%
%   output ----------------------------------------------------------------
%
%       o MSE_F_fold      : (1 x K), Mean Squared Error computed for each value of k averaged over the number of folds.
%       o NMSE_F_fold     : (1 x K), Normalized Mean Squared Error computed for each value of k averaged over the number of folds.
%       o R2_F_fold       : (1 x K), Coefficient of Determination computed for each value of k averaged over the number of folds.
%       o AIC_F_fold      : (1 x K), Mean AIC Scores computed for each value of k averaged over the number of folds.
%       o BIC_F_fold      : (1 x K), Mean BIC Scores computed for each value of k averaged over the number of folds.
%       o std_MSE_F_fold  : (1 x K), Standard Deviation of Mean Squared Error computed for each value of k.
%       o std_NMSE_F_fold : (1 x K), Standard Deviation of Normalized Mean Squared Error computed for each value of k.
%       o std_R2_F_fold   : (1 x K), Standard Deviation of Coefficient of Determination computed for each value of k averaged over the number of folds.
%       o std_AIC_F_fold  : (1 x K), Standard Deviation of AIC Scores computed for each value of k averaged over the number of folds.
%       o std_BIC_F_fold  : (1 x K), Standard Deviation of BIC Scores computed for each value of k averaged over the number of folds.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mean of metrics
MSE_F_fold      = zeros(1, length(k_range));
NMSE_F_fold     = zeros(1, length(k_range));
R2_F_fold = zeros(1, length(k_range));
AIC_F_fold      = zeros(1, length(k_range));
BIC_F_fold      = zeros(1, length(k_range));

% Std of metrics
std_MSE_F_fold      = zeros(1, length(k_range));
std_NMSE_F_fold     = zeros(1, length(k_range));
std_R2_F_fold = zeros(1, length(k_range));
std_AIC_F_fold      = zeros(1, length(k_range));
std_BIC_F_fold      = zeros(1, length(k_range));

MSE = zeros(F_fold,1);
NMSE = zeros(F_fold,1);
Rsquared = zeros(F_fold,1);
AIC = zeros(F_fold,1);
BIC = zeros(F_fold,1);

[N,~] = size(X);
[P,~] = size(y);
in = 1:N;
out = N+1:(N+P);


for k=1:length(k_range)
    params.k = k_range(k);
    for f=1:F_fold
        [X_train, y_train, X_test, y_test] = split_regression_data(X, y, valid_ratio);
        Xtr=[X_train; y_train];
        Xts=[X_test; y_test];
        [Priors, Mu, Sigma, ~, ~, ~, ~] = my_gmmEM(Xtr, params);
        [yest, ~] = my_gmr(Priors, Mu, Sigma, X_test, in, out);
        [MSE(f), NMSE(f), Rsquared(f)] = my_regression_metrics(yest, y_test);
        [AIC(f), BIC(f)] =  gmm_metrics(Xtr, Priors, Mu, Sigma, params.cov_type);
    end
    
    MSE_F_fold(1,k) = mean(MSE);
    NMSE_F_fold(1,k) = mean(NMSE);
    R2_F_fold(1,k) = mean(Rsquared);
    AIC_F_fold(1,k) = mean(AIC);
    BIC_F_fold(1,k) = mean(BIC);
    std_MSE_F_fold(1,k) = std(MSE);
    std_NMSE_F_fold(1,k) = std(NMSE);
    std_R2_F_fold(1,k) = std(Rsquared);
    std_AIC_F_fold(1,k) = std(AIC);
    std_BIC_F_fold(1,k) = std(BIC);
end

end

