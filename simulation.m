% Simulation to test the derived equations
% 12.11.2018
% Hilkka Hännikäinen

close all;
clear all;

% add matlab_bgl and directory containing the data to the path

load G.mat % lead field matrix
G = double(G); % for more precision
load mesh_ds.mat % information
load anat_label_name_mne.mat % names of anatomical areas
load anat_labels_mne.mat % anatomical areas (index of source point)
load ignored_sp.mat % indices of ignored points to construct the lead field matrix
load included_sp.mat % indices of included points to construct the lead field matrix
load flip.mat % orientation of the dipoles

% set some parameters
% s = number of sensors (dimension of sensor space)
% q = number of dipoles (dimension of down sampled source space)
[s,q] = size(G); 
dim = size(mesh_ds.p,1); % number of original source points
snr = 10; % signal to noise ratio

% consider only magnetometers
magnetometers = 1:3:305;
s = length(magnetometers);
G = G(magnetometers,:);

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
    anat_labels{1,i} = anat_labels{1,i} + 1;
    [~,~,indx] = intersect(anat_labels{1,i}, lh_p_ss);
    anat_ds.parcels{i,1} = indx;
end

% right hemisphere
for i=2:2:length(anat_labels)
    anat_labels{1,i} = anat_labels{1,i} + 1; % python data
    [~,~,indx] = intersect(anat_labels{1,i}, rh_p_ss);
    anat_ds.parcels{i,1} = indx+length(lh_p_ss);
end
clear lh rh lh_p_ss rh_p_ss indx

areas = {'superiortemporal-rh', 'superiortemporal-lh'}; % names of the ROIs
p = size(areas, 2); % number of ROIs
% extract information concerning the chosen ROIs
for j=1:p
    for i=1:68
        % compare two strings
        is_same = strcmp(areas{j}, strtrim(anat_label_name(i,:)));
        if (is_same)
            parcels{j} = anat_ds.parcels{i,1}; % extract parcel corresponding to ROIs
            sp_parcels(j) = size(parcels{j},1); % number of source points in ROI
            roi_ind(j) = i; % index of the region
            break
        end   
    end
end

for j=1:p
    index{j} = included_sp(parcels{j}); % indices of the points in original source space
    crds{j} = mesh_ds.p(index{j},:); % extraxt coordinates
    cntr{j} = min(crds{j}) + (max(crds{j}-min(crds{j})))/2; % calculate the geometric centroid
    % find the index of a source point closest to the centroid coordinates
    dst = vecnorm(crds{j}-repmat(cntr{j}, sp_parcels(j), 1), 2, 2);
    [~, ind_min] = min(dst);
    cntr_crds(j,:) = crds{j}(ind_min,:);
    id(j) = index{j}(ind_min);
    id_ds(j) = parcels{j}(ind_min);
    clear dst
    clear ind_min
end

for j=1:p
    not_in_ROI_ss{j} = setdiff(1:dim, index{j}); % points outside the ROI in source space
    not_in_ROI_ds{j} = setdiff(1:q, parcels{j}); % points outside the ROI in down sampled source space
end

% formulate penalty matrices using euclidean distance
phi = 10*ones(1,p); % weight for orientation
lambda = 1*ones(1,p); % weight or distance
for j=1:p
    Q_pen = zeros(sp_parcels(j));
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

f = zeros(q,p); % orientation matrix for the chosen ROIs
for j=1:p
   f(parcels{j},j) = flip{roi_ind(j)};
end

radii = [0.03, 0.03]; % for radii used as a variances
mu = [0,0]; % expectation values
GW = get_dist_matrix(mesh_ds); % distance matrix between the points in original source space
GW = GW(included_sp, included_sp);
% define weight for each point using normal distribution
L = zeros(q,p);
for j=1:p
    L(:,j) = source_spread(GW, id_ds(j), radii(j));
    L(not_in_ROI_ds{j},j) = 0; % zero out points not belonging to ROI
end
L = normc(L); % normalization
L = L.*f; % multiply elementwise with orientation matrix

% formulate penalty matrices using euclidean distance
phi = 10*ones(1,p); % weight for orientation
lambda = 1*ones(1,p); % weight or distance
for j=1:p
    Q_pen = zeros(sp_parcels(j));
    for i=1:1:length(index{j})
        for k=i:1:length(index{j})
          Q_pen(i,k) = phi(j)*mesh_ds.nn(index{j}(i),:)*mesh_ds.nn(index{j}(k),:)'*...
          exp(-lambda(j)*(sqrt(sum((mesh_ds.p(index{j}(i),:) - mesh_ds.p(index{j}(k),:)).^2))));
%           Q_pen(i,k) = exp(-lambda(j)*(sqrt(sum((mesh_ds.p(index{j}(i),:) - mesh_ds.p(index{j}(k),:)).^2))));
        end
    end
   Q_tmp = Q_pen'-diag(diag(Q_pen'));
   Qp{j} = Q_pen + Q_tmp;
   invQp{j} = inv(Qp{j}); % make inverse of the penalty matrix
end
clear Q_pen Q_tmp

% generate data
T = 100; % number of timepoints
A = eye(p); % evolution matrix
u = zeros(p, T); % signal for the activity of the area
cov_u = 1e-18.*eye(p); % covariance of the normal distribution at time t=0
mu_u = zeros(1,p); % mean of the normal distribution at time t=0
u(:,1) = mvnrnd(mu_u, cov_u); % initialize using multivariate normal distribution
S = 1e-18.*eye(p); % state noise covariance
noise_s = sqrt(S)*randn(p,T); % state noise
for i=2:T
    u(:,i) = A*u(:,i-1)+noise_s(:,i);    
end
R = 1e-20.*eye(q); % dipole noise covariance
q_t = L*u + sqrt(R)*randn(q,T); % dipole activity
Q = 1e-24.*eye(s); % measurement noise covariance
y = G*q_t + sqrt(Q)*randn(s,T); % simulated measurement data
sigma = G*R*G'+Q; % total noise covariance
invSigma = inv(sigma);

L_prev = zeros(q, p);
% create initial quess for L: define weight for each point using normal distribution
radii_est = [0.07, 0.07];
L_est = zeros(q,p);
for j=1:p
    L_est(:,j) = source_spread(GW, id_ds(j), radii_est(j));
    L_est(not_in_ROI_ds{j},j) = 0;
end
L_est = normc(L_est);
L_est = L_est.*f;

% initialization of variables
% pred = predicted values
% filt = filtered values
% smoo = smoothed values
u_pred = zeros(p, T);
u_filt = zeros(p, T);
u_smoo = zeros(p, T);
P_pred = cellmat(1, T, p, p, 0);
P_filt = cellmat(1, T, p, p, 0);
P_smoo = cellmat(1, T, p, p, 0);
P_lag = cellmat(1, T, p, p, 0); % the lag-one covariance smoother
J = cellmat(1, T, p, p, 0);

% initialization step
u_smoo(:,1) = mu_u;
P_smoo{:,1} = cov_u;

max_iter = 100; % maximum nummber of iterations
min_step = 1e-9; % minimum step size allowed
loglikelihoods = zeros(1, max_iter); % log-likelihoods
% difference between current and previous parameter value
norms_diff = zeros(1, max_iter);

% estimation of parameters
for k=1:max_iter
    L_prev = L_est;
    u_filt(:,1) = u_smoo(:,1); % "update" of mu_0
    P_filt{1} = P_smoo{1}; % "update" of covU_0
    % Kalman filtering
    GL_tmp = G*L_prev;
    for i=2:T
        u_pred(:,i)=A*u_filt(:,i-1);
        P_pred{i}=A*P_filt{i-1}*A' + S;
        K = P_pred{i}*GL_tmp'/(GL_tmp*P_pred{i}*GL_tmp'+sigma);
        u_filt(:,i) = u_pred(:,i)+K*(y(:,i)-GL_tmp*u_pred(:,i));
        P_filt{i} = (eye(p)-K*GL_tmp)*P_pred{i};
    end
    % initializing smoother
    u_smoo(:,T) = u_filt(:,T);
    P_smoo{T} = P_filt{T};
    P_lag{T} = (eye(p)-K*GL_tmp)*A*P_smoo{T-1};
    for i=T:-1:2 % backward smoothing
        J{i-1} = P_filt{i-1}*A'/P_pred{i};
        u_smoo(:,i-1) = u_filt(:,i-1)+J{i-1}*(u_smoo(:,i)-u_pred(:,i));
        P_smoo{i-1} = P_filt{i-1}+J{i-1}*(P_smoo{i}-P_pred{i})*J{i-1}';
    end
    for i=T:-1:3 % lag one covariance smoother
        P_lag{i-1} = P_filt{i-1}*J{i-2}+J{i-1}*(P_lag{i}-A*P_filt{i-1})*J{i-2}';
    end
    % calculate gradient
    grad = gradient(y, u_smoo, P_smoo, G, L_prev, invSigma, invQp, parcels);
    for j=1:p
        grad(not_in_ROI_ds{j},j) = 0; % zero out points not belonging to ROIs
    end
    % set parameters for btls
    alpha = 0.4;
    beta = 0.8;
    step = 1;
    % initialize probabilities
    [curr_prob, ~] = log_likelihood(u_smoo, mu_u, cov_u, A, S, P_smoo, P_lag, ...
        y, G, L_prev, sigma, invSigma, Qp, invQp, parcels);
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
    % L_step = abs(L_step);
    L_step = normc(L_step);
    [next_prob, ~] = log_likelihood(u_smoo, mu_u, cov_u, A, S, P_smoo, P_lag, ...
        y, G, L_step, sigma, invSigma, Qp, invQp, parcels);
    % find the step size using backtracking line search
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
        [next_prob, ~] = log_likelihood(u_smoo, mu_u, cov_u, A, S, P_smoo, P_lag, ...
            y, G, L_step, sigma, invSigma, Qp, invQp, parcels);
    end
    loglikelihoods(k) = next_prob; % store th log-likelihood
    L_est = L_step; % update the estimate
    norms_diff(k) = norm(L_est-L_prev);
end

% to visualize the results
figure
hold on
plot(u_pred(1,:))
plot(u_filt(1,:))
plot(u_smoo(1,:))
plot(u(1,:))
legend('prediction', 'filtering', 'smoothing', 'signal')
