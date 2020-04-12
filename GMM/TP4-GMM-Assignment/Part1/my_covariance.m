function [ Sigma ] = my_covariance( X, X_bar, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o X_bar : (N x 1), an Nx1 matrix corresponding to mean of data X
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%

% Auxiliary Variable
[N, M] = size(X);

% Output Variable
Sigma = zeros(N, N);

if type == "full"
    Sigma = ((X-X_bar)*transpose(X- X_bar))/(M-1);
elseif type == "diag"
    Sigma = diag(diag(((X-X_bar)*transpose(X-X_bar))/(M-1)));
elseif type == "iso"
    sum = 0;
    for i=1:M
        sum = sum + norm(X(:,i)-X_bar)^2;
    end
    sum = (1/(N*M))*sum;
    sum = repmat(sum, N,1);
    Sigma = diag(sum);
end


end

