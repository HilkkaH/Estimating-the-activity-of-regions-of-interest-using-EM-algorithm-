% code to test the validity of the maximumlikelihood method
% 6.11.2018
% Hilkka Hännikäinen

close all;
clear all;

% add matlab_bgl and directory containing data to the path

load G.mat % the lead field matrix
load bad_ind.mat % index of the bad channel
G(bad_ind, :) = []; % exclude the bad channel
load Y_aud.mat % the data
Y(bad_ind, :) = [];
Y(306:end,:) = []; % exclude other than MEG channels
load C.mat % the measurement noise covariance matrix
load J_aud.mat % the minimum norm estimate of dipole activity
load mesh_ds.mat % coordinates, normal vectors and triangles
load anat_label_name_mne.mat % names of anatomical areas
load anat_labels_mne.mat % anatomical areas (index of source point)
load ignored_sp.mat % indices of ignored points to construct the lead field matrix
load included_sp.mat % indices of included points to construct the lead field matrix
load flip.mat % vector containing 1s and -1s
% u's are 2 x 421 matrices so that the first row corresponds to right
load u_mean_st.mat
load u_mean_flip_st.mat
load u_pca_st.mat

[U, S, V] = svd(C); % C is the noise covariance matrix
tol = 1e-12; % check the eigen/singular values to determine tol
diagS = diag(S);
sel = find(diagS>tol.*diagS(1));
P = diag(1./sqrt(diag(S(sel,sel))))*U(:,sel)'; % prewhitening matrix
% whiten the data
Y = P*Y;
G = P*G;
clear C
% set some parameters
% s = number of sensors (dimension of sensor space)
% q = number of dipoles (dimension of down sampled source space)
[s,q] = size(G);
Y = Y(1:s, :);
dim = size(mesh_ds.p,1); % number of original source points

% consider only magnetometers
magnetometers = 1:3:305;
s = length(magnetometers);
G = G(magnetometers,:);
Y = Y(magnetometers,:);
Q = eye(s);

% ectract the parcel data
tmp = load('src_left');
lh = getfield(tmp, 'src');
tmp = load('src_right');
rh = getfield(tmp, 'src');
clear tmp;

anatomical.names = cellstr(anat_label_name);
lh_p_ss = lh.vertno + 1;
rh_p_ss = rh.vertno + 1;

% left hemispehere
for i=1:2:length(anat_labels)
    anat_labels{1,i} = anat_labels{1,i} + 1; % python data
    [~,~,indx] = intersect(anat_labels{1,i}, lh_p_ss);
    anat_ds.parcels{i,1} = indx;
end

% right hemisphere
for i=2:2:length(anat_labels)
    anat_labels{1,i} = anat_labels{1,i} + 1; % python data
    [~,~,indx] = intersect(anat_labels{1,i}, rh_p_ss);
    anat_ds.parcels{i,1} = indx+length(lh_p_ss);
end
clear indx lh rh lh_p_ss rh_p_ss

% extract area of interest, its coordinates and centroid
% put the right hemisphere first
areas = {'superiortemporal-rh', 'superiortemporal-lh'}; % names of the regions
p = size(areas, 2); % number of ROIs
% extract information concerning the chosen ROIs
for j=1:p
for i=1:68
    is_same = strcmp(areas{j}, strtrim(anat_label_name(i,:)));
    if (is_same)
        roi_ind(j) = i;
        parcels{j} = anat_ds.parcels{i,1}; % extract parcel corresponding to region
        break
    end   
end
end
for j=1:p
    dip = abs(J(parcels{j},:)); % extract dipole activity corresponding to the region
    maximums = max(dip, [], 2); % take maximum of each row (dipole)
    [~, ind] = max(maximums); % take the index of source having the maximum activity
    index{j} = included_sp(parcels{j});
    cntr(j) = index{j}(ind);
    not_in_ROI_ss{j} = setdiff(1:dim, index{j}); % points outside the ROI in source space
    not_in_ROI_ds{j} = setdiff(1:q, parcels{j}); % points outside the ROI in down sampled source space
end
clear dip maximums ind % clear out unnecessary variables

% formulate penalty matrices using euclidean distance
phi = 10*ones(1,p); % weight for orientation
lambda = 1*ones(1,p); % weight or distance
for j=1:p
Q_pen = zeros(size(parcels{j},1));
for i=1:1:length(index{j})
    for k=i:1:length(index{j})
     Q_pen(i,k) = phi(j)*mesh_ds.nn(index{j}(i),:)*mesh_ds.nn(index{j}(k),:)'*...
     exp(-lambda(j)*(sqrt(sum((mesh_ds.p(index{j}(i),:) - mesh_ds.p(index{j}(k),:)).^2))));
    end
end
   Q_tmp = Q_pen'-diag(diag(Q_pen'));
   Qp{j} = Q_pen + Q_tmp;
   invQp{j} = inv(Qp{j}); % make inverse of the penalty matrix
end
clear Q_pen Q_tmp

f = zeros(q,p); % "orientation matrix"
for j=1:p
   f(parcels{j},j) = flip{roi_ind(j)};
end
clear roi_ind

L_prev = zeros(q, p);
% create initial quess for L: define weight for each point using normal distribution
radii_est = 0.05.*ones(1,p);
mu_est = zeros(1,p);
GW = get_dist_matrix(mesh_ds);
GW = GW(included_sp, included_sp);
L_est = zeros(q,p);
for j=1:p
    L_est(:,j) = source_spread(GW, cntr(j), radii_est(j));
    L_est(not_in_ROI_ds{j},j) = 0;
end
L_est = normc(L_est);
L_est = L_est.*f; % elementwise multiplication to take the orientation into account
L_init = L_est; % to compare the initialisation and the result of estimation

% estimate the signal covariance
max_data = max(max(Y));
max_GL = max(max(G*L_est));
cov_u = (max_data/max_GL)^2*eye(p);
clear max_data max_GL
R = 0; % 

max_iterations = 1000; % maximum amount of iterations allowed
likelihoods = zeros(1, max_iterations); % to store the likelihoods
norms_diff = zeros(1, max_iterations); % to compare the difference between current and previous estimate
min_step = 1e-9; % minimum step size allowed
crt = 1e-9; % stopping criterion

for k=1:max_iterations
    L_prev = L_est;
    % calculate the gradient
    grad = gradient_max(Y, cov_u, G, L_prev, Q, R, invQp, parcels);
    for j=1:p
        grad(not_in_ROI_ds{j},j) = 0; % zero out points not belonging to ROI
    end
    % set parameters for btls
    alpha = 0.4;
    beta = 0.8;
    step = 1;
    % initialize probabilities
    curr_prob = objective_function(Y, cov_u, G, L_prev, Q, R, invQp, parcels);
    L_step = L_prev-step.*grad; % prepare the step
    for j=1:p
        L_step(not_in_ROI_ds{j}, j) = 0; % zero out points not belonging to ROIs
    end
    % to restrict the orientation
    L_sign = sign(L_step); % take the sign of the weights
    L_sign = L_sign == f; % compare with the orientation vector
    for j=1:p
        incorrect =  find(L_sign(:,j) == 0); % find indices of incorrect ones
        L_step(incorrect, j) = -1.*L_step(incorrect, j); % correct the signs
    end
    L_step = normc(L_step);
    next_prob = objective_function(Y, cov_u, G, L_step, Q, R, invQp, parcels);
    % find the step size
    while next_prob > curr_prob - alpha*step*norm(grad, 'fro') & step > min_step
        step = beta*step;
        L_step = L_prev-step.*grad;
        for j=1:p
            L_step(not_in_ROI_ds{j},j) = 0;
        end
        L_sign = sign(L_step); % take the sign of the weights
        L_sign = L_sign == f; % compare with the orientation vector
        for j=1:p
            incorrect =  find(L_sign(:,j) == 0); % find indices of incorrect ones
            L_step(incorrect, j) = -1.*L_step(incorrect, j); % correct the signs
        end
        L_step = normc(L_step);
        next_prob = objective_function(Y, cov_u, G, L_step, Q, R, invQp, parcels);
    end
    likelihoods(k) = next_prob;
    L_est = L_step; % update the estimate
    norms_diff(k) = norm(L_est-L_prev)/norm(L_prev);
    if k > 1
        if norms_diff(k) < crt
            break
        end
    end
end

% calculate minimumnormestimate for the region activity
u_ml = lsqminnorm(L_est, J);

% compare the results to ones obtained with other methods
% for left hemisphere
figure
hold on
plot(u_mean_st(2,:))
plot(u_mean_flip_st(2,:))         
plot(u_pca_st(2,:))
plot(u_ml(2,:))
title('Left hemisphere, superiortemporal')
legend('mean', 'flipped mean', 'PCA', 'ML')

% for right hemisphere
figure
hold on
plot(u_mean_st(1,:))
plot(u_mean_flip_st(1,:))
plot(u_pca_st(1,:))
plot(u_ml(1,:))
title ('Right hemisphere, superiortemporal')
legend('mean', 'flipped mean', 'PCA', 'ML')
