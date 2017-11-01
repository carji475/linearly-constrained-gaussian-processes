function [L,g] = margLik_diag(sg,rx,ry,y)
% returns the negative log marginal likelihood and gradient for the diagonal 
% covariance function
%
% INPUTS
%   sg - hyperparameters (signal variance, length scale, noise std)  
%   rx, ry - matrices storing the differences between x- and y-components
%   y - measurements
%
% OUTPUTS
%   L - negative log marginal likelihood   
%   g - gradient
%
try
    %% extract hyperparameters
    sigma_f=sg(1); 
    l=sg(2);
    sigma_n=sg(3);
    
    %% # measurements
    n_obs=length(y)/2;
    
    %% Gram matix
    K = [sigma_f^2*exp(-0.5*(rx.^2+ry.^2)/l^2) zeros(n_obs)
        zeros(n_obs) sigma_f^2*exp(-0.5*(rx.^2+ry.^2)/l^2)]+sigma_n^2*eye(2*n_obs);
    K = K+2*abs(min([eig(K); 0]))*eye(size(K)); % increase numerical stability
    
    %% calculate function value and gradient
    alpha = K\y;
    
    L = 0.5*y'*alpha+0.5*sum(log(eig(K)))+log(2*pi); % - log ML
    
    dKdsigma_f1 = [2*sigma_f*exp(-0.5*(rx.^2+ry.^2)/l^2) zeros(n_obs)
        zeros(n_obs) 2*sigma_f*exp(-0.5*(rx.^2+ry.^2)/l^2)];
    dKdl1 = [sigma_f^2/l^3*(rx.^2+ry.^2).*exp(-0.5*(rx.^2+ry.^2)/l^2) zeros(n_obs)
        zeros(n_obs) sigma_f^2/l^3*(rx.^2+ry.^2).*exp(-0.5*(rx.^2+ry.^2)/l^2)];
    dKdsigma_n = 2*sigma_n*eye(2*n_obs);
    
    beta = alpha*alpha'-K\eye(2*n_obs);
    
    g1 = -0.5*sum(sum(beta.*dKdsigma_f1));
    g2 = -0.5*sum(sum(beta.*dKdl1));
    g3 = -0.5*sum(sum(beta.*dKdsigma_n));
    g  = [g1 g2 g3]'; % gradient
catch % case something goes wrong
    L=nan;
    g=nan(5,1);
end
end