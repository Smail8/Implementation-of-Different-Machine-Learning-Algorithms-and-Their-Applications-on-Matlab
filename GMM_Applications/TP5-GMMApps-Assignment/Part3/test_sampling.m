%% DO NOT MODIFY THIS UNLESS YOU ARE ON YOUR OWN COMPUTER
addpath(genpath("..\..\..\ML_toolbox-master"))
addpath(genpath("~/Repositories/ML_toolbox/")) % TODO CHANGE FOR
%WINDOWS LOCATION

addpath(genpath("../utils"))
addpath(genpath("../Part1"))

clear; 
close all; 
clc;

dataset_path = '../../TP5-GMMApps-Datasets/';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Sampling from 2D Dataset
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nbPoints = 1000;
%% ADD CODE HERE
% Load a dataset, and train GMM model and sample nbPoints points from it
data = halfkernel();
X = data(:,1:2)';
Y = data(:,3)';

params.cov_type = 'full';
params.k = 6;
params.max_iter = 500;
params.d_type = 'L2';
params.init = 'plus';
params.max_iter_init = 100;

[models] = learn_gmm(X, Y, params);
XNew = draw_samples(models, nbPoints);

% END CODE

% plot both the original data and the sampled ones
figure('Name', 'Original dataset')
dotsize = 12;
ax1 = subplot(1,2,1);
scatter(X(1,:), X(2,:), dotsize);
title('Original Data')
ax2 = subplot(1,2,2);
scatter(XNew(1,:), XNew(2,:), dotsize);
title('Sampled Data')
linkaxes([ax1,ax2],'xy')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         2) Sampling from high-dimensional data
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_data = csvread('mnist_train.csv', 1, 0);  
nbSubSamples = 2000;
idx = randperm(size(train_data, 1), nbSubSamples);
train_data = train_data(idx,:);

% extract the data
Xtrain = train_data(:,2:end)';
Ytrain = train_data(:,1)';

plot_digits(Xtrain)

p = 40; % in cases you need this, use this value

%% ADD CODE HERE
[eigenvectors, eigenvalues, Mean] = compute_pca(Xtrain) ;
[Ap, Xp] = project_pca(Xtrain, Mean, eigenvectors, p) ;

params.cov_type = 'full';
params.k = 20;
params.max_iter = 500;
params.d_type = 'L2';
params.init = 'plus';
params.max_iter_init = 100;

models = learn_gmm(Xp, Ytrain, params);

XNew = draw_samples(models, nbSubSamples);

XHat = reconstruct_pca(XNew, Ap, Mean) ;

% plot the reconstructed

plot_digits(XHat)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Sampling from a GMM per class
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ADD CODE HERE
% we can also train different models of GMM for each class of digits
params.cov_type = 'full';
params.k = 20;
params.max_iter_init = 100;
params.max_iter = 500;
params.d_type = 'L2';
params.init = 'plus';

[eigenvectors, eigenvalues, Mean] = compute_pca(Xtrain) ;
[Ap, Xp] = project_pca(Xtrain, Mean, eigenvectors, p) ;

models = learn_gmm(Xp, Ytrain, params);
% sample fron the GMM that are trained on this specific digit
class = 5;
X = sample_per_class(models, class, 100);

% reconstruct the images from the PCA
XHat = reconstruct_pca(X, Ap, Mean);
%% END CODE

% plot the reconstructed
plot_digits(XHat)





