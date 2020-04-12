function [ exp_var, cum_var, p ] = explained_variance( L, Var )
%EXPLAINED_VARIANCE Function that returns the optimal p given a desired
%   explained variance. The student should convert the Eigenvalue matrix 
%   to a vector and visualize the values as a 2D plot.
%   input -----------------------------------------------------------------
%   
%       o L      : (N x N), Diagonal Matrix composed of lambda_i 
%       o Var    : (1 x 1), Desired Variance to be explained
%  
%   output ----------------------------------------------------------------
%
%       o exp_var  : (N x 1) vector of explained variance
%       o cum_var  : (N x 1) vector of cumulative explained variance
%       o p        : optimal principal components given desired Var


% ====================== Implement Eq. 8 Here ====================== 
exp_var = diag(L)./sum(diag(L));
% ====================== Implement Eq. 9 Here ====================== 
cum_var = cumsum(exp_var);
% ====================== Implement Eq. 10 Here ====================== 
p = min(find(cum_var > Var));

% Visualize/Plot Explained Variance from Eigenvalues
plot(1:length(cum_var), cum_var);
xlabel('Eigenvectors Index');
ylabel('Cumulative Variance Explained');

end

