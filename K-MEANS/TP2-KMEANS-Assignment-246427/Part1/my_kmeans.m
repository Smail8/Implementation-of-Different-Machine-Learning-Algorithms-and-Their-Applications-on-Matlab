function [labels, Mu, Mu_init, iter] =  my_kmeans(X,K,init,type,MaxIter,plot_iter)
%MY_KMEANS Implementation of the k-means algorithm
%   for clustering.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o K        : (int), chosen K clusters
%       o init     : (string), type of initialization {'random','uniform'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%       o plot_iter: (bool), boolean to plot iterations or not (only works with 2d)
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a vector with predicted labels labels \in {1,..,k} 
%                   corresponding to the k-clusters.
%       o Mu       : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N 
%       o Mu_init  : (N x k), same as above, corresponds to the centroids used
%                            to initialize the algorithm
%       o iter     : (int), iteration where algorithm stopped
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
[N, M] = size(X);
d_i    = zeros(K,M);
k_i    = zeros(K,M);
r_i    = zeros(K,M);
if plot_iter == [];plot_iter = 0;end

% Auxiliary Variable
[N, M] = size(X);
if plot_iter == [];plot_iter = 0;end

% Output Variables
labels  = zeros(1,M);
Mu      = zeros(N, K);
Mu_init = zeros(N, K);
Mu_old = zeros(N, K);
ppc = zeros(1, K);
iter      = 0;

% Step 1. Mu Initialization
Mu_init = kmeans_init(X, K, init); 
Mu = Mu_init;

%%%%%%%%%%%%%%%%%         TEMPLATE CODE      %%%%%%%%%%%%%%%%
% Visualize Initial Centroids if N=2 and plot_iter active
colors     = hsv(K);
if (N==2 && plot_iter)
    options.title       = sprintf('Initial Mu with <%s> method', init);
    ml_plot_data(X',options); hold on;
    ml_plot_centroid(Mu_init',colors);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iter = 0;
while true

    %%%%% Implement K-Means Algorithm HERE %%%%%    
    r_i    = zeros(K,M);
    ppc = zeros(1, K);
    % Step 2. Distances from X to Mu
    d_i = my_distX2Mu(X, Mu, type);
    
	% Step 3. Assignment Step: Mu Responsability
	% Equation 5 and 6
    [S , k_i] = sort(d_i, 1);
    
    for i=1:M 
        for k=2:K-1
            if(S(1, i) == S(k, i))
                if (ppc(1) > ppc(k))
                    k_i(1,i) = k_i(k, i);
                end
            end
        end
       cluster_id = k_i(1,i);
       r_i(cluster_id, i) = 1;
       ppc(cluster_id) = ppc(cluster_id) + 1;
       labels(i) = cluster_id;
    end
    
	% Step 4. Update Step: Recompute Mu	
    % Equation 7
    
    Mu_old = Mu;
    for k=1:K
         S_1 = zeros(N, 1);
         S_2 = 0;
         for i=1:M
            S_1 = S_1 + r_i(k, i).*X(:,i);
            S_2 = S_2 + r_i(k,i);
         end
         if(S_2 ~= 0)
            Mu(:, k) = S_1./S_2;
         end
    end
    
	% Check for stopping conditions (Mu stabilization or MaxIter)      
    if (isequal(Mu, Mu_old) || (iter == MaxIter))
        break;
    end
    
    %%%%%%%%%%%%%%%%%         TEMPLATE CODE      %%%%%%%%%%%%%%%%       
    if (N==2 && iter == 1 && plot_iter)
        options.labels      = labels;
        options.title       = sprintf('Mu and labels after 1st iter');
        ml_plot_data(X',options); hold on;
        ml_plot_centroid(Mu',colors);
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    iter = iter+1;
    
    if(min(ppc) == 0 || min(ppc) == 1)
        Mu_init = kmeans_init(X, K, init);
        Mu = Mu_init;
    end
    
end


%%%%%%%%%%%   TEMPLATE CODE %%%%%%%%%%%%%%%
if (N==2 && plot_iter)
    options.labels      = labels;
    options.class_names = {};
    options.title       = sprintf('Mu and labels after %d iter', iter);
    ml_plot_data(X',options); hold on;    
    ml_plot_centroid(Mu',colors);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
   