clear all; close all;

%% measurements noise
sigma_n=1e-4; 

%% predictive mesh
nx=20; ny=nx; n_pred=nx*ny;
xx = linspace(0,4,nx);
yy = linspace(0,4,ny);
[X,Y] = meshgrid(xx,yy);

%% true function
a = 0.01;
Fx = @(X,Y) -X.*exp(-a*X.*Y).*cos(X.*Y) + a*X.*exp(-a*X.*Y).*sin(X.*Y);
Fy = @(X,Y) Y.*exp(-a*X.*Y).*cos(X.*Y) - a*Y.*exp(-a*X.*Y).*sin(X.*Y);
ytrue=[Fx(X,Y) Fy(X,Y)]; ytrue=ytrue(:);

%% optimisation options
opti_options = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter-detailed',...
    'GradObj','on','TolFun',1e-8);
theta = [rand(2,1); sigma_n];

%% parameter setttings
nrRepeat = 50; % # repetitions (for reduced impact of randomness)
nrConst  = [25 50 75 100 150 200 300 400]; % # artificial observations explored

%% allocate for result data collection
errDiagObs_all=zeros(nrRepeat,length(nrConst));
errCust_all=zeros(nrRepeat,1);
errDiag_all=zeros(nrRepeat,1);

for zz=1:nrRepeat
    rng shuffle % random seed
    
    %% measurements
    n_obs = 50; % number of observations
    obs_points = 4*rand(n_obs,2); % observe in random points
    % true values
    obs_values_x = Fx(obs_points(:,1),obs_points(:,2)); 
    obs_values_y = Fy(obs_points(:,1),obs_points(:,2));
    y = [obs_values_x; obs_values_y];
    y = mvnrnd(y',sigma_n^2*ones(1,2*n_obs))'; % add noise
    
    %% predefined matrices for r=x-x' to keep syntax uncluttered
    rxg = repmat(obs_points(:,1),1,n_obs)-repmat(obs_points(:,1)',n_obs,1);
    ryg = repmat(obs_points(:,2),1,n_obs)-repmat(obs_points(:,2)',n_obs,1);
    rxp = repmat(obs_points(:,1),1,nx*ny)-repmat(X(:)',n_obs,1);
    ryp = repmat(obs_points(:,2),1,nx*ny)-repmat(Y(:)',n_obs,1);
    
    %% Customised covariance function
    % call the optimising routine
    theta_opt_cust = fminunc(@(thetas) margLik_cust(thetas,rxg,ryg,y),theta,opti_options);
    
    sigma_f_cust    = theta_opt_cust(1); % signal variance
    l_cust          = theta_opt_cust(2); % length scale
    sigma_n_cust    = theta_opt_cust(3); % noise level
    
    % Gram matrix
    K_cust = sigma_f_cust^2*[(1-ryg.^2/l_cust^2)/l_cust^2 rxg.*ryg/l_cust^4
        rxg.*ryg/l_cust^4 (1-rxg.^2/l_cust^2)/l_cust^2].*repmat(exp(-0.5*(rxg.^2+ryg.^2)/l_cust^2),2,2)+sigma_n_cust^2*eye(2*n_obs);
    
    % predictive cross covariance
    k_cust=sigma_f_cust^2*[(1-ryp.^2/l_cust^2)/l_cust^2 rxp.*ryp/l_cust^4
        rxp.*ryp/l_cust^4 (1-rxp.^2/l_cust^2)/l_cust^2].*repmat(exp(-0.5*(rxp.^2+ryp.^2)/l_cust^2),2,2);
    
    % cross covariance
    L_cust          = chol(K_cust + 2*abs(min([eig(K_cust); 0]))*eye(length(K_cust)))'; % cholesky factorisation with numerical stabilisation
    alpha_cust      = L_cust'\(L_cust\y);
    meanCust        = k_cust'*alpha_cust;
    errCust_all(zz) = rms(ytrue-meanCust); % store rms error
    
    %% Diagonal covariance function
    theta_opt_diag = fminunc(@(thetas) margLik_diag(thetas,rxg,ryg,y),theta,opti_options);
    
    sigma_f_diag = theta_opt_diag(1); % signal variance;
    l_diag       = theta_opt_diag(2); % length scale
    sigma_n_diag = theta_opt_diag(3); % noise level
    
    % Gram matrix
    K_diag = [sigma_f_diag^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag^2) zeros(n_obs)
        zeros(n_obs) sigma_f_diag^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag^2)]+sigma_n_diag^2*eye(2*n_obs);
    
    % cross covariance
    k_diag = [sigma_f_diag^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag^2) zeros(n_obs,nx*ny)
        zeros(n_obs,nx*ny) sigma_f_diag^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag^2)];
    
    % predict
    L_diag          = chol(K_diag + 2*abs(min([eig(K_diag); 0]))*eye(length(K_diag)))'; % cholesky factorisation with numerical stabilisation
    alpha_cust      = L_cust'\(L_cust\y);
    alpha_diag      = L_diag'\(L_diag\y);
    meanDiag        = k_diag'*alpha_diag;
    errDiag_all(zz) = rms(ytrue-meanDiag);
    
    %% Diagonal with constraint observations in predictive points
    % constraints
    indices=randperm(nrConst(end)); % randomly ordered indices of all mesh points
    for qq=1:length(nrConst)
        Xc=X(indices(1:nrConst(qq))); Yc=Y(indices(1:nrConst(qq))); % choose a subset of the mesh points
        
        n_const=length(Xc(:)); % (=nrConst(qq))
        
        rxgc = repmat(obs_points(:,1),1,n_const)-repmat(Xc(:)',n_obs,1);
        rygc = repmat(obs_points(:,2),1,n_const)-repmat(Yc(:)',n_obs,1);
        rxcp = repmat(Xc(:),1,nx*ny)-repmat(X(:)',n_const,1);
        rycp = repmat(Yc(:),1,nx*ny)-repmat(Y(:)',n_const,1);
        rxcc = repmat(Xc(:),1,n_const)-repmat(Xc(:)',n_const,1);
        rycc = repmat(Yc(:),1,n_const)-repmat(Yc(:)',n_const,1);
        
        theta_opt_diag_obs = theta_opt_diag; % use the same hyperparameters as in the diagonal case
        
        sigma_f_diag_obs=theta_opt_diag_obs(1); % signal variance;
        l_diag_obs = theta_opt_diag_obs(2);     % length scale
        sigma_n_diag_obs=theta_opt_diag_obs(3); % noise level
        
        % Gram matrix
        K_diag_obs = [sigma_f_diag_obs^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag_obs^2)+sigma_n_diag_obs^2*eye(n_obs) zeros(n_obs) -rxgc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)
            zeros(n_obs) sigma_f_diag_obs^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag_obs^2)+sigma_n_diag_obs^2*eye(n_obs) -rygc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)
            [-rxgc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)]' [-rygc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)]' (1-rxcc.^2/l_diag_obs^2)/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcc.^2+rycc.^2)/l_diag_obs^2)+...
            (1-rycc.^2/l_diag_obs^2)/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcc.^2+rycc.^2)/l_diag_obs^2)];
        
        % cross covariance
        k_diag_obs=[sigma_f_diag_obs^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag_obs^2) zeros(n_obs,nx*ny)
            zeros(n_obs,nx*ny) sigma_f_diag_obs^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag_obs^2)
            rxcp/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcp.^2+rycp.^2)/l_diag_obs^2) rycp/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcp.^2+rycp.^2)/l_diag_obs^2)];
        
        % predict
        K_diag_obs                = K_diag_obs + 2*abs(min([eig(K_diag_obs); 0]))*eye(length(K_diag_obs));        
        meanDiagObs               = k_diag_obs'*(K_diag_obs\[y; zeros(n_const,1)]);
        errDiagObs_all(zz,qq)     = rms(ytrue-meanDiagObs);
    end
end
%% calculate the average errors
errCust     = mean(errCust_all);
errDiag     = mean(errDiag_all);
errDiagObs  = mean(errDiagObs_all);

%% plot the results
loglog(nrConst,errCust*ones(1,length(nrConst)),'r-','LineWidth',2); hold on;
loglog(nrConst,errDiag*ones(1,length(nrConst)),'c-','LineWidth',2)
loglog(nrConst,errDiagObs,'vk-','LineWidth',2)
grid on
legend('Our approach','Diagonal','Artificial obs')
xlabel('$N_c$','Interpreter','latex','FontSize',14)
ylabel('$e_{\mathrm{rms}}$','Interpreter','latex','FontSize',14)