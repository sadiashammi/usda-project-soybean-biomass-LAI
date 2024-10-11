%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Author:    Weiwei Xie
%  Created:   09.20.2024
%  For any questions and/or comments for this code/paper, please feel free
%  to contact Dr. Weiwei Xie
%  Email: wx38@msstate.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

load('R3_Biomass.mat')

N = size(x, 1);
num_fold = N;
[x_folds, y_folds] = crossval(data, N, num_fold);

y_test_linear = []; y_test_svm = []; y_test_rf = [];
for i = 1 : size(x_folds{1}, 2)
    i
    for k = 1 : num_fold
        x_test = x_folds{k};
        y_test = y_folds{k};
        x_train = []; y_train = [];
        for j = 1 : num_fold
            if (j == k)
                continue
            end
            x_train = [x_train; x_folds{j}];
            y_train = [y_train; y_folds{j}];
        end
        x_train = x_train(:, i);
        x_test = x_test(:,i);
        mdl = fitlm(x_train, y_train);
        ypred = predict(mdl, x_test);
        y_test_linear = [y_test_linear; [ypred y_test]];
    
        mdl = fitrsvm(x_train,y_train,'KernelFunction','gaussian','Standardize',true, 'GapTolerance', 1e-2);
        ypred = predict(mdl, x_test);
        y_test_svm = [y_test_svm; [ypred y_test]];
        
        mdl = fitrensemble(x_train,y_train, 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', templateTree('MaxNumSplits',10, 'MergeLeaves','on','Prune','off','PruneCriterion','mse','SplitCriterion','mse'));
%         mdl = fitrensemble(x_train,y_train, 'Method', 'Bag', 'NumLearningCycles', 100);
        ypred = predict(mdl, x_test);
        y_test_rf = [y_test_rf; [ypred y_test]];
    
    end
    
    ypred = y_test_linear(:, 1);  y_test = y_test_linear(:, 2);
    r2 = 1 - sum((ypred - y_test).^2) / sum((y_test - mean(y_test)).^2);
    rmse = sqrt(mean((ypred - y_test).^2));
    nrmse = rmse / mean(ypred);
    cv_linear_results(i,:)  = [r2 rmse];
    
    ypred = y_test_svm(:, 1);  y_test = y_test_svm(:, 2);
    rmse = sqrt(mean((ypred - y_test).^2));
    nrmse = rmse / mean(ypred);
    r2 = 1 - sum((ypred - y_test).^2) / sum((y_test - mean(y_test)).^2);
    cv_svm_results(i,:)  = [r2 rmse];
    
    ypred = y_test_rf(:, 1);  y_test = y_test_rf(:, 2);
    rmse = sqrt(mean((ypred - y_test).^2));
    nrmse = rmse / mean(ypred);
    r2 = 1 - sum((ypred - y_test).^2) / sum((y_test - mean(y_test)).^2);
    adj_r2 = 1 - (1-r2) * (length(y_test)-1) / (length(y_test) - size(x_test, 2) - 1);
    cv_rf_results(i,:) = [r2 rmse];
end



