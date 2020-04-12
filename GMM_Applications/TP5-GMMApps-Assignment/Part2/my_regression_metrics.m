function [MSE, NMSE, Rsquared] = my_regression_metrics( yest, y )
%MY_REGRESSION_METRICS Computes the metrics (MSE, NMSE, R squared) for 
%   regression evaluation
%
%   input -----------------------------------------------------------------
%   
%       o yest  : (P x M), representing the estimated outputs of P-dimension
%       of the regressor corresponding to the M points of the dataset
%       o y     : (P x M), representing the M continuous labels of the M 
%       points. Each label has P dimensions.
%
%   output ----------------------------------------------------------------
%
%       o MSE       : (1 x 1), Mean Squared Error
%       o NMSE      : (1 x 1), Normalized Mean Squared Error
%       o R squared : (1 x 1), Coefficent of determination
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary Variables
[~, M] = size(y);
MSE = 0;
var_y = 0;
R_numerator = 0;
R_denominator1 = 0;
R_denominator2 = 0;
Mu = mean(y, 2);
Mu_est = mean(yest, 2);

for i=1:M
    MSE = MSE + (yest(:,i) - y(:,i)).^2;
    var_y = var_y + (y(:,i) - Mu).^2;
    R_numerator = R_numerator + ((y(:,i)-Mu)*(yest(:,i)-Mu_est));
    R_denominator1 = R_denominator1 + (y(:,i)-Mu).^2;
    R_denominator2 = R_denominator2 + (yest(:,i)-Mu_est).^2;
end

MSE = MSE/M;
NMSE = MSE/(var_y/(M-1));
Rsquared = (R_numerator.^2)/(R_denominator1*R_denominator2);

end

