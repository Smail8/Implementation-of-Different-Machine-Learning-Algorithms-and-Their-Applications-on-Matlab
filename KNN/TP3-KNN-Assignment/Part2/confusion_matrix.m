  function [C] =  confusion_matrix(y_test, y_est)
%CONFUSION_MATRIX Implementation of confusion matrix 
%   for classification results.
%   input -----------------------------------------------------------------
%
%       o y_test    : (1 x M), a vector with true labels y \in {1,2} 
%                        corresponding to X_test.
%       o y_est     : (1 x M), a vector with estimated labels y \in {1,2} 
%                        corresponding to X_test.
%
%   output ----------------------------------------------------------------
%       o C          : (2 x 2), 2x2 matrix of |TP & FN|
%                                             |FP & TN|.
%        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = zeros(2,2);
[~, M_test] = size(y_test);

for i=1:M_test
    if y_test(1, i) == 2
        if y_est(1, i) == 2
            C(1, 1) = C(1, 1) + 1;
        elseif y_est(1, i) == 1
            C(1, 2) = C(1, 2) + 1;
        end
    elseif y_test(1, i) == 1
        if y_est(1, i) == 2
            C(2, 1) = C(2, 1) + 1;
        elseif y_est(1, i) == 1
            C(2, 2) = C(2, 2) + 1;
        end
    end
end

end

