function [gradient] = gradient(y, u, P, G, L, invSigma, invQpen, par_index)
% Function calculates the gradient of the negative log-likelihood function
% with respect to L
% y = the data
% u = the smoothed values of the signal
% G = the lead field matrix
% L = the weight matrix
% sigma = the total noise covariance in the measurement equation
% invQpen = inverses of the penalty matrices for L
% par_index = indices of dipoles in ROIs in downsampled source space

[p, T] = size(u); % number ofregions and timepoints
[s, q] = size(G); % number of sensors and dipoles
grad = zeros(q,p);
% calculate sums needed in the gradient
sum00 = zeros(p);
sumY = zeros(s);
sumYU = zeros(s,p);
sumUY = zeros(p,s);
for i = 1:T
    sum00 = sum00 + u(:,i)*u(:,i)'+P{i};
    sumY = sumY + y(:,i)*y(:,i)';
    sumYU = sumYU + y(:,i)*u(:,i)';
    sumUY = sumUY + u(:,i)*y(:,i)';
end
% gradient for L
grad = -G'*invSigma*(sumYU-G*L*sum00)-G'*invSigma'*(sumUY'-G*L*sum00);
% add the penalty for each region to the gradient
for j=1:p
    penalty = 2*invQpen{j}*L(par_index{j}, j);
    grad(par_index{j},j) = grad(par_index{j},j) + penalty;
end
gradient = grad;
end