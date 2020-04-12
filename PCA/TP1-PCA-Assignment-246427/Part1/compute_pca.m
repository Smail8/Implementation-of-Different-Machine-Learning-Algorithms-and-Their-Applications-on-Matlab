function [ V, L, Mu ] = compute_pca( X )
%MY_PCA Step-by-step implementation of Principal Component Analysis
%   In this function, the student should implement the Principal Component 
%   Algorithm following Eq.1, 2 and 3 of Assignment 1.
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o V      : (M x M), Eigenvectors of Covariance Matrix.
%       o L      : (M x M), Eigenvalues of Covariance Matrix
%       o Mu     : (N x 1), Mean Vector of Dataset

% Auxiliary variables
[N, M] = size(X);

% Output variables
V  = zeros(M,M);
L  = zeros(M,M);
Mu = zeros(N,1);

% ====================== Implement Eq. 1 Here ====================== 
Mu = mean(X')';
X = bsxfun(@minus, X, Mu);

% ====================== Implement Eq.2 Here ======================
C = X*X'./(M-1);
% ====================== Implement Eq.3 Here ======================
[V, L] = svd(C);
% =================== Sort Eigenvectors wrt. EigenValues ==========


end

