function [gradient] = gradient_max(y, covU, G, L, Q, R, invQpen, par_index)
% function calculating the gradient of L
% input arguments:
% y = the data
% covU = covriance of the ROIs
% G = the lead field matrix
% L = the weight matrix
% R = the dipole noise covariance
% invQpen = inveres of the penalty matrices
% par_index = index of the source points in ROIs in downsampled source
% space
% pen_parameters = gammadistribution parameters for dipole noise
% covariances
[~,p] = size(L); % number of regions
[s,T] = size(y); % number of sensors and timepoints
% calculation of the covariance matrix
delta = G*L*covU*(G*L)'+Q;
mtrx = bsxfun(@times, diag(R), G'); % to speed up the computation
delta = delta + G*mtrx;
clear mtrx
invDelta = inv(delta); % inverse of the covariance
sumY = zeros(s,s); % sum of y*y^T
for i = 1:T
   sumY = sumY + y(:,i)*y(:,i)';
end
subtraction = (invDelta-invDelta*sumY*invDelta); % auxiliary variable
% gradient with respect to L
grad = 2*G'*subtraction'*G*L*covU;
% add penalty to the gradient
for j=1:p
    grad_pen = 2*invQpen{j}*L(par_index{j},j);
    grad(par_index{j},j) = grad(par_index{j},j) + grad_pen;
end
gradient = grad;
end