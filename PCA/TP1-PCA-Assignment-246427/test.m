clc
clear all
close all

A = [0 1 2 
    3 4 5 
    6 7 8
    9 10 11];

Mu = mean(A);

Ap = bsxfun(@minus, A, mean(A')');

C = Ap*Ap'/2;

[U, S, V] = svd(Ap);

