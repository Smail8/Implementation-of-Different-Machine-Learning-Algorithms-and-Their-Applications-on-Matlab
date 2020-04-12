function [RSS_curve, AIC_curve, BIC_curve] =  kmeans_eval(X, K_range,  repeats, init, type, MaxIter)
%KMEANS_EVAL Implementation of the k-means evaluation with clustering
%metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%       o init     : (string), type of initialization {'random','uniform','plus'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%
%   output ----------------------------------------------------------------
%       o RSS_curve  : (1 X K_range), RSS values for each value of K \in K_range
%       o AIC_curve  : (1 X K_range), AIC values for each value of K \in K_range
%       o BIC_curve  : (1 X K_range), BIC values for each value of K \in K_range
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RSS_curve = zeros(1, length(K_range));
AIC_curve = zeros(1, length(K_range));
BIC_curve = zeros(1, length(K_range));

K = length(K_range);

for i=1:K
   r=1;
   while(r <= repeats)
       [labels, Mu, ~, ~] = my_kmeans(X, K_range(i), init, type, MaxIter, []);
       [RSS, AIC, BIC] = my_metrics(X, labels, Mu);
       RSS_curve(i) = RSS_curve(i) + RSS;
       AIC_curve(i) = AIC_curve(i) + AIC;
       BIC_curve(i) = BIC_curve(i) + BIC;
       r = r + 1 ;
   end
   RSS_curve(i) = RSS_curve(i) / repeats;
   AIC_curve(i) = AIC_curve(i) / repeats;
   BIC_curve(i) = BIC_curve(i) / repeats;
end


end
