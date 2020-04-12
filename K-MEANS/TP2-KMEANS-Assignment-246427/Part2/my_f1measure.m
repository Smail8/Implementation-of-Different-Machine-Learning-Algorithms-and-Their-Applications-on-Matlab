function [F1_overall, P, R, F1] =  my_f1measure(cluster_labels, class_labels)
%MY_F1MEASURE Computes the f1-measure for semi-supervised clustering
%
%   input -----------------------------------------------------------------
%   
%       o class_labels     : (N x 1),  N-dimensional vector with true class
%                                       labels for each data point
%       o cluster_labels   : (N x 1),  N-dimensional vector with predicted 
%                                       cluster labels for each data point
%   output ----------------------------------------------------------------
%
%       o F1_overall      : (1 x 1)     f1-measure for the clustered labels
%       o P               : (nClusters x nClasses)  Precision values
%       o R               : (nClusters x nClasses)  Recall values
%       o F1              : (nClusters x nClasses)  F1 values
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
M         = length(class_labels);
true_K    = unique(class_labels);
found_K   = unique(cluster_labels);
nClasses  = length(true_K);
nClusters = length(found_K);
K = zeros(nClusters);
C = zeros(nClasses);
n = zeros(nClusters, nClasses);
% Output Variables
P = zeros(nClusters, nClasses);
R = zeros(nClusters, nClasses);
F1 = zeros(nClusters, nClasses);
F1_overall = 0;

for i=1:nClusters
    for j=1:nClasses
        n(i,j) = length(find(cluster_labels == i & class_labels == j));
    end
end

for i=1:nClusters
    K(i) = length(find(cluster_labels == i));
   
end

for i=1:nClasses
    C(i) = length(find(class_labels == i));
end

for i=1:nClusters
    for j=1:nClasses
        P(i, j) = n(i, j) / K(i);
        R(i, j) = n(i, j) / C(j);
        if R(i, j) + P(i, j) ~= 0
            F1(i, j) = (2*R(i, j)*P(i, j))/(R(i, j) + P(i, j));
        end
    end
end

for i=1:nClasses
   F1_overall = F1_overall + (C(i)/M)*max(F1(:, i)); 
end
 
end
