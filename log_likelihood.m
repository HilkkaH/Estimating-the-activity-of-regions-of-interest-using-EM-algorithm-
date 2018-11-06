function [loglikelihood, A_est] = log_likelihood(u, muU, covU, A, S, P, P_lag, y, G, L, sigma, invSigma, Qpen, invQpen, par_index)
% function calculates the negative log-likelihood
% u = smoothed values of the signal
% muU = mean for the distribution of u_0
% covU = covariance of the distribution of u_0
% A = evolution matrix
% S = covariance of the noise in the evolution equation
% P = smoothed values of the covariance of u
% P_lag = lag one covariance
% y = obtained data
% G = the lead field matrix
% L = the weight matrix
% sigma = the noise covariance in the measurement equation
[p, T] = size(u); % p = number of regions, T = number of time points
[s, q] = size(G); % s = number of sensors, q = number of dipoles 
% constant term for the initial probability of u
term1 = p*log(2*pi) + sum(log(diag(chol(covU)))) + (u(:,1)-muU')'/covU*(u(:,1)-muU');
% initialize the sums
sum00 = u(:,1)*u(:,1)'+P{1};
sum01 = zeros(p);
sum11 = zeros(p);
sumY = y(:,1)*y(:,1)';
sumYU = y(:,1)*u(:,1)';
sumUY = u(:,1)*y(:,1)';
% calculate the sums appearing in the equation for the likelihood
for i=2:T
    sum00 = sum00 + u(:,i)*u(:,i)'+P{i};
    sum01 = sum01 + u(:,i)*u(:,i-1)'+P_lag{i};
    sum11 = sum11 + u(:,i-1)*u(:,i-1)'+P{i-1};
    sumY = sumY + y(:,i)*y(:,i)';
    sumYU = sumYU + y(:,i)*u(:,i)';
    sumUY = sumUY + u(:,i)*y(:,i)';
end
% calculate the constant terms
term2 = T*p*log(2*pi) + T*sum(log(diag(chol(S)))) + ...
    trace((sum00-sum01*A'-A*sum01'+A*sum11*A')\S);
term3 = T*s*log(2*pi) + T*sum(log(diag(chol(sigma)))) + ...
    trace(invSigma*(sumY-sumYU*L'*G'-G*L*sumUY-G*L*sum00*L'*G'));
% calculate the penalty term
term4 = 0;
for j=1:p
   term4 = term4 + size(par_index{j},2)*log(2*pi) + ...
       sum(log(diag(chol(Qpen{j})))) + trace(invQpen{j}*L(par_index{j},j)*L(par_index{j},j)');
end
loglikelihood = term1 + term2 + term3 + term4;
A_est = sum01/sum11; % update the estimate for A
% update the estimate for process noise
% S_est = 1/T*(sum00-sum01*A'-A*sum01+A*sum11*A');
% update the estimate for combined measurement noise
% sigma_est = 1/T*(sumY-sumYU*L'*G'-G*L*sumUY+G*L*sum00*L'*G');
end