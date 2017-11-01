clear all; close all;

%% measurements noise
sigma_n=1e-4; 

%% predictive mesh
nx=20; ny=nx; n_pred=nx*ny;
xx = linspace(0,4,nx);
yy = linspace(0,4,ny);
[X,Y] = meshgrid(xx,yy);

%% constraint observations
nxc=20; nyc=nxc; n_const=nxc*nyc;
xxc = linspace(0,4,nxc);
yyc = linspace(0,4,nyc);
[Xc,Yc] = meshgrid(xxc,yyc);

%% true function
a=0.01;
Fx = @(X,Y) -X.*exp(-a*X.*Y).*cos(X.*Y) + a*X.*exp(-a*X.*Y).*sin(X.*Y);
Fy = @(X,Y) Y.*exp(-a*X.*Y).*cos(X.*Y) - a*Y.*exp(-a*X.*Y).*sin(X.*Y);
ytrue=[Fx(X,Y) Fy(X,Y)]; ytrue=ytrue(:);

%% measured points
rng shuffle
n_obs = 50; % number of observations
obs_points = 4*rand(n_obs,2); % observe in random points
obs_values_x = Fx(obs_points(:,1),obs_points(:,2));
obs_values_y = Fy(obs_points(:,1),obs_points(:,2));
y = [obs_values_x; obs_values_y];
y = mvnrnd(y',sigma_n^2*ones(1,2*n_obs))';

%% predefined structures to keep syntax uncluttered
rxg = repmat(obs_points(:,1),1,n_obs)-repmat(obs_points(:,1)',n_obs,1);
ryg = repmat(obs_points(:,2),1,n_obs)-repmat(obs_points(:,2)',n_obs,1);
rxp = repmat(obs_points(:,1),1,nx*ny)-repmat(X(:)',n_obs,1);
ryp = repmat(obs_points(:,2),1,nx*ny)-repmat(Y(:)',n_obs,1);
rxgc = repmat(obs_points(:,1),1,nxc*nyc)-repmat(Xc(:)',n_obs,1);
rygc = repmat(obs_points(:,2),1,nxc*nyc)-repmat(Yc(:)',n_obs,1);
rxcp = repmat(Xc(:),1,nx*ny)-repmat(X(:)',nxc*nyc,1);
rycp = repmat(Yc(:),1,nx*ny)-repmat(Y(:)',nxc*nyc,1);
rxcc = repmat(Xc(:),1,nxc*nyc)-repmat(Xc(:)',nxc*nyc,1);
rycc = repmat(Yc(:),1,nxc*nyc)-repmat(Yc(:)',nxc*nyc,1);

%% optimisation options
opti_options = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter-detailed',...
    'GradObj','on','TolFun',1e-8);
theta = [rand(2,1); sigma_n];

%% Customised covariance
% call the optimising routine
theta_opt_cust = fminunc(@(thetas) margLik_cust(thetas,rxg,ryg,y),theta,opti_options);

sigma_f_cust = theta_opt_cust(1); % signal variance;
l_cust       = theta_opt_cust(2); % length scale
sigma_n_cust = theta_opt_cust(3); % noise level

% Gram matrix
K_cust = sigma_f_cust^2*[(1-ryg.^2/l_cust^2)/l_cust^2 rxg.*ryg/l_cust^4
    rxg.*ryg/l_cust^4 (1-rxg.^2/l_cust^2)/l_cust^2].*repmat(exp(-0.5*(rxg.^2+ryg.^2)/l_cust^2),2,2)+sigma_n_cust^2*eye(2*n_obs);

% predictive
k_cust = sigma_f_cust^2*[(1-ryp.^2/l_cust^2)/l_cust^2 rxp.*ryp/l_cust^4
    rxp.*ryp/l_cust^4 (1-rxp.^2/l_cust^2)/l_cust^2].*repmat(exp(-0.5*(rxp.^2+ryp.^2)/l_cust^2),2,2);

% predict
L_cust      = chol(K_cust + 2*abs(min([eig(K_cust); 0]))*eye(length(K_cust)))';
alpha_cust  = L_cust'\(L_cust\y);
meanCust    = k_cust'*alpha_cust;
errCust     = rms(ytrue-meanCust);

%% Diagonal
theta_opt_diag = fminunc(@(thetas) margLik_diag(thetas,rxg,ryg,y),theta,opti_options);

sigma_f_diag = theta_opt_diag(1); % signal variance;
l_diag       = theta_opt_diag(2); % length scale
sigma_n_diag = theta_opt_diag(3); % noise level

% Gram matrix
K_diag = [sigma_f_diag^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag^2) zeros(n_obs)
    zeros(n_obs) sigma_f_diag^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag^2)]+sigma_n_diag^2*eye(2*n_obs);

% predictive
k_diag = [sigma_f_diag^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag^2) zeros(n_obs,nx*ny)
    zeros(n_obs,nx*ny) sigma_f_diag^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag^2)];

% predict
L_diag      = chol(K_diag + 2*abs(min([eig(K_diag); 0]))*eye(length(K_diag)))';
alpha_diag  = L_diag'\(L_diag\y);
meanDiag    = k_diag'*alpha_diag;
errDiag     = rms(ytrue-meanDiag);

%% Diagonal with constraint observations in predictive points
theta_opt_diag_obs = theta_opt_diag; % use the same hyperparameters as in the diagonal case

sigma_f_diag_obs = theta_opt_diag_obs(1); % signal variance;
l_diag_obs       = theta_opt_diag_obs(2); % length scale
sigma_n_diag_obs = theta_opt_diag_obs(3); % noise level

% Gram matrix
K_diag_obs = [sigma_f_diag_obs^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag_obs^2)+sigma_n_diag_obs^2*eye(n_obs) zeros(n_obs) -rxgc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)
    zeros(n_obs) sigma_f_diag_obs^2*exp(-0.5*(rxg.^2+ryg.^2)/l_diag_obs^2)+sigma_n_diag_obs^2*eye(n_obs) -rygc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)
    [-rxgc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)]' [-rygc/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxgc.^2+rygc.^2)/l_diag_obs^2)]' (1-rxcc.^2/l_diag_obs^2)/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcc.^2+rycc.^2)/l_diag_obs^2)+...
    (1-rycc.^2/l_diag_obs^2)/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcc.^2+rycc.^2)/l_diag_obs^2)];

% predictive
k_diag_obs = [sigma_f_diag_obs^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag_obs^2) zeros(n_obs,nx*ny)
    zeros(n_obs,nx*ny) sigma_f_diag_obs^2*exp(-0.5*(rxp.^2+ryp.^2)/l_diag_obs^2)
    rxcp/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcp.^2+rycp.^2)/l_diag_obs^2) rycp/l_diag_obs^2*sigma_f_diag_obs^2.*exp(-0.5*(rxcp.^2+rycp.^2)/l_diag_obs^2)];

% predict
K_diag_obs  = K_diag_obs + 2*abs(min([eig(K_diag_obs); 0]))*eye(length(K_diag_obs));
meanDiagObs = k_diag_obs'*(K_diag_obs\[y; zeros(n_const,1)]);
errDiagObs  = rms(ytrue-meanDiagObs);

%% plots
scalefactor=10;
%%% 1 - true field with measurements
% meas field
subplot(1,4,1)
quiver(obs_points(:,1),obs_points(:,2),y(1:length(y)/2)/scalefactor,y(length(y)/2+1:end)/scalefactor,0,'color',[1 0 0])
% true field
hold on
quiver(X,Y,Fx(X,Y)/scalefactor,Fy(X,Y)/scalefactor,0,'color',[0.7 0.7 0.7])
axis([0 4 0 4])
% legend('Measurements','True')
xlabel('$x_1$','Interpreter','latex','FontSize',14)
ylabel('$x_2$','Interpreter','latex','FontSize',14)

Fxx=Fx(X,Y)/scalefactor; Fxx=Fxx(:);
Fyy=Fy(X,Y)/scalefactor; Fyy=Fyy(:);

%%% 2 - div-free kernel with measurements
subplot(1,4,2)
quiver(X(:),Y(:),-(meanCust(1:length(meanCust)/2)/scalefactor-Fxx),-(meanCust(length(meanCust)/2+1:end)/scalefactor-Fyy),0,'color',[0 0 0])
axis([0 4 0 4])
xlabel('$x_1$','Interpreter','latex','FontSize',14)
ylabel('$x_2$','Interpreter','latex','FontSize',14)
title(['Our approach: $e_\mathrm{rms}=$' num2str(errCust)],'Interpreter','latex') 

%%% 3 - diagonal kernel with measurements
subplot(1,4,3)
quiver(X(:),Y(:),-meanDiag(1:length(meanDiag)/2)/scalefactor+Fxx,-meanDiag(length(meanDiag)/2+1:end)/scalefactor+Fyy,0,'color',[0 0 0])
axis([0 4 0 4])
xlabel('$x_1$','Interpreter','latex','FontSize',14)
ylabel('$x_2$','Interpreter','latex','FontSize',14)
title(['Diagonal: $e_\mathrm{rms}=$' num2str(errDiag)],'Interpreter','latex') 

%%% 4 - diagonal kernel+pseudo-observ with measurements
subplot(1,4,4)
quiver(X(:),Y(:),-meanDiagObs(1:length(meanDiagObs)/2)/scalefactor+Fxx,-meanDiagObs(length(meanDiagObs)/2+1:end)/scalefactor+Fyy,0,'color',[0 0 0])
axis([0 4 0 4])
xlabel('$x_1$','Interpreter','latex','FontSize',14)
ylabel('$x_2$','Interpreter','latex','FontSize',14)
title(['Artificial obs: $e_\mathrm{rms}=$' num2str(errDiagObs)],'Interpreter','latex')