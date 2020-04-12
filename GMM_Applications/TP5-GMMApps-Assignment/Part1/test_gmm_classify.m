%% DO NOT MODIFY THIS UNLESS YOU ARE ON YOUR OWN COMPUTER
addpath(genpath("..\..\..\ML_toolbox-master"))
addpath(genpath("~/Repositories/ML_toolbox/")) % TODO CHANGE FOR
%WINDOWS LOCATION

addpath(genpath("../utils"))

clear; 
close all; 
clc;

dataset_path = '../../TP5-GMMApps-Datasets/';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) GMM-Classification 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ADD CODE HERE
% Load a dataset, and train GMM models
% You can use anY of the dataset functions provided in the 
% dataset folder in utils

data = halfkernel();
X = data(:,1:2)';
Y = data(:,3)';

split_ratio = 0.4;

[~, M] = size(X);
rand_idx = randperm(M);
X = X(:,rand_idx);
Y = Y(rand_idx);
Y = Y + 1;

[Xtrain, Ytrain, Xtest, Ytest] = split_data(X, Y, split_ratio);

params.cov_type = 'full';
params.k = 4;
params.max_iter = 500;
params.d_type = 'L2';
params.init = 'plus';
params.max_iter_init = 100;

[models] = learn_gmm(Xtrain, Ytrain, params);






%% END CODE

% visualize the GMM fitting
figure('Name', 'Original dataset')
dotsize = 12;
scatter(data(:,1), data(:,2), dotsize, data(:,3)); axis equal;

% DisplaY contours for each class
for c = 1:length(unique(Y))
    ml_plot_gmm_pdf(Xtrain, models(c).Priors, models(c).Mu, models(c).Sigma)
    hold off 
end

%% ADD CODE HERE
% Perform classification on the testing set
classes = unique(Ytrain);
Y_est = ML_discriminant_rule(Xtest, classes, models, params.k);



%% END CODE 

% Compute AccuracY
acc =  my_accuracy(Ytest, Y_est);

% visualize it
figure('Name', 'Classification with GMM')
ax1 = subplot(1,2,1);
dotsize = 12;
scatter(data(:,1), data(:,2), dotsize, data(:,3)); axis equal;
title('Original Data');

% Plot decision boundarY
ax2 = subplot(1,2,2);
plot_boundaries(Xtrain, Ytrain, Xtest, Ytest, Y_est,  models, params.k);
linkaxes([ax1,ax2],'xy')