function [S] = gower_similarity(X1,X2,data_type, rk)
%GOWER_SIMILARITY Compute the Gower similarity between X1 and X2
%
%   input -----------------------------------------------------------------
%       o X1 : (N x 1) First sample point
%       o X2 : (N x 1) Second sample point
%       o data_type : {N x 1}, a boolean cell array with true when
%                     feature is continuous
%       o rk : (N x 1) The range of values for continuous data 
%
%   output ----------------------------------------------------------------
%      o S : The similarity measure between two samples

% Auxiliary variables
N = size(X1,1);
Si = zeros(N,1);

% ADD CODE HERE: For each feature i of N calculate the Gower similarity. At
% the end sum it to calculate the similarity between the tzo samples.
% HINT: The formula varies depending if the feature is catagorical or
% continuous.
sum = 0;
for i=1:N
    if data_type{i} == 1
        Si(i) = 1 - (abs(X1(i) - X2(i))/rk(i));
    else
        if X1(i) == X2(i)
            Si(i) = 1;
        else
            Si(i) = 0;
        end
    end
    sum = sum + Si(i);
end

S = sum/N;



% END CODE

