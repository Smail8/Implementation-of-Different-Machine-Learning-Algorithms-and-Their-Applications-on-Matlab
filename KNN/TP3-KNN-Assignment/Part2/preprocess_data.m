function [X, Y, rk] = preprocess_data(table_data, ratio, data_type)
%PREPROCESS_DATA Preprocess the data in the adult dataset
%
%   input -----------------------------------------------------------------
%
%       o table_data    : (M x N), a cell array containing mixed
%                         categorical and continuous values
%       o ratio : The pourcentage of M samples to extract 
%
%   output ----------------------------------------------------------------
%       o X : (N-1, M*ratio) Data extracted from the table where
%             categorical values are converted to integer values
%       o Y : (1, M*ratio) The label of the data to classify. Values are 1
%             or 2
%       o rk : (N x 1) The range of values for continuous data (will be 0
%               if the data are categorical)

% Auxiliary Variables
rk = zeros(size(table_data,2),1);

% ADD CODE HERE: Convert features data to numerical values. If the data are 
% categorical first convert them to int values. If the data are continuous 
% store the range in rk.
% HINT: Type of feature data (continuous or categorical) is stored in
% data_type which is boolean cell array (true if continuous). Only select
% the samples based on idx array. Be careful with the input and output 
% dimensions.

[M, N] = size(table_data);
M_r = floor(M*ratio);
X = zeros(N-1, M_r);
Y = zeros(1, M_r);

P = randperm(M, M_r);

for i=1:M_r
    dataset(i,:) = table_data(P(i),:);
end

for i=1:N-1
    if data_type{i} == true
        X(i,:) = table2array(dataset(:,i));
        rk(i,1) = range(X(i,:));
    else
        [X(i,:),~] = grp2idx(table2array(dataset(:,i)));
    end
end

Y(1, :) = grp2idx(table2array(dataset(:,N)));


% END CODE
end

