clear all; addpath(genpath('../'))

% parameters
nrdata_train = 5e2;
nrdata_test  = 1e3;

% data set
load dataSet14 % non-disturbed 3D-measurements of the magnetic field

pos=data_obj{2}.pos; pos=pos(1:end-1,:); % last row=NaN
mag=data_obj{2}.mag; mag=mag(1:end-1,:);

% number of data points
nrdata=size(pos,1);

% optimisation options
opti_options = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter-detailed',...
    'GradObj','on','TolFun',1e-8);
theta = -10*rand(1,4); % start guess for hyperparameter selection

% allocate
nrRepeat=50; nrCurl=round(logspace(1,3,8));
errDiagObs_all=zeros(nrRepeat,length(nrCurl));
errCurl_all=zeros(nrRepeat,1);
errDiag_all=zeros(nrRepeat,1);
for zz=1:nrRepeat
    % randomly pick train and test data
    indices = randperm(nrdata);
    
    % training data
    Xfield = pos(indices(1:nrdata_train),:);
    yfield = mag(indices(1:nrdata_train),:);
    yfield = yfield(:);
    
    % test data
    Xpred = pos(indices(nrdata_train+1:nrdata_train+nrdata_test),:);
    ytrue = mag(indices(nrdata_train+1:nrdata_train+nrdata_test),:);
    ytrue = ytrue(:);
    
    %% diagonal regression (without curl observations)
    theta_opt_diag = fminunc(@(thetas) margLik(thetas,Xfield,zeros(0,3),yfield,'diag'),theta,opti_options);
    
    [K_diag,kpred_diag] = diag_kernel(theta_opt_diag,Xfield,zeros(0,3),Xpred);
    L_diag              = chol(K_diag + 2*abs(min([eig(K_diag); 0]))*eye(length(K_diag)))';
    alpha_diag          = L_diag'\(L_diag\yfield);
    meanDiag            = kpred_diag*alpha_diag;
    errDiag_all(zz)     = rms(ytrue-meanDiag);
    
    %% diagonal regression (with curl observations)
    for qq=1:length(nrCurl)
        % curl observations (in (some) points of predictions)
        nrcurl       = nrCurl(qq);
        indices_curl = randperm(nrdata_test,nrcurl);
        Xcurl        = Xpred(indices_curl,:);
        
        theta_opt_diag_obs = theta_opt_diag; % use the diagonal parameters
        
        [K_diag_obs,kpred_diag_obs] = diag_kernel(theta_opt_diag_obs,Xfield,Xcurl,Xpred);
        K_diag_obs              = K_diag_obs + 2*abs(min([eig(K_diag_obs); 0]))*eye(length(K_diag_obs));
        meanDiagObs             = kpred_diag_obs*(K_diag_obs\[yfield; zeros(3*nrcurl,1)]);
        errDiagObs_all(zz,qq)   = rms(ytrue-meanDiagObs);
    end
    
    %% curl regression
    theta_opt_curl = fminunc(@(thetas) margLik(thetas,Xfield,zeros(0,3),yfield,'curl'),theta,opti_options);
    
    K_curl = curl_free_kernel(theta_opt_curl,Xfield,Xfield);
    kpred_curl = curl_free_kernel(theta_opt_curl,Xpred,Xfield);
    L_curl = chol(K_curl + 2*abs(min([eig(K_curl); 0]))*eye(length(K_curl)))';
    alpha_curl = L_curl'\(L_curl\yfield);
    meanCurl = kpred_curl*alpha_curl;
    errCurl_all(zz)=rms(ytrue-meanCurl);
end
%% average errors
errCurl=mean(errCurl_all);
errDiag=mean(errDiag_all);
errDiagObs=mean(errDiagObs_all,1);

%% plot
semilogy(nrCurl,errCurl*ones(1,length(nrCurl)),'r-','LineWidth',2); hold on;
semilogy(nrCurl,errDiag*ones(1,length(nrCurl)),'c-','LineWidth',2)
semilogy(nrCurl,errDiagObs,'vk-','LineWidth',2)
grid on
legend('Our approach','Diagonal','Artificial obs')
xlabel('$N_c$','Interpreter','latex','FontSize',14)
ylabel('$e_{\mathrm{rms}}$','Interpreter','latex','FontSize',14)