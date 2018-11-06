function [neg_llh] = objective_function(y, covU, G, L, Q, R, invQpen, par_index, pen_parameters)
% calculate the negative log-likelihood to obtain the data
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
[s,T] = size(y); % number of sensors and time points
[q,p] = size(L); % number of dipoles and regions
neg_llh = 0;
% calculate covariance matrix
cov_mat = G*L*covU*(G*L)'+Q;
mtrx = bsxfun(@times, diag(R), G'); % to speed up the computation
cov_mat = cov_mat + G*mtrx;
% calculate the log-likelihood as sum of likelihoods for distinct time
% points
for i=1:T
    neg_llh = neg_llh+y(:,i)'/cov_mat*y(:,i);
end
% calculate the constant term
const = T*s*log(2*pi)+T*sum(log(diag(chol(cov_mat))));
% calculate the likelihood of penalty terms
for j=1:p
    pen(j) = L(par_index{j},j)'*invQpen{j}*L(par_index{j},j);
end

neg_llh = neg_llh + const + sum(pen);
end