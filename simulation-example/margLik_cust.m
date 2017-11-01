function [L,g] = margLik_cust(sg,rx,ry,y)
% returns the negative log marginal likelihood and gradient for the
% customised covariance function
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
    
    %% Gram matrix
    K=sigma_f^2*[(1-ry.^2/l^2)/l^2 rx.*ry/l^4
        rx.*ry/l^4 (1-rx.^2/l^2)/l^2].*repmat(exp(-0.5*(rx.^2+ry.^2)/l^2),2,2)+sigma_n^2*eye(2*n_obs);
    K = K+2*abs(min([eig(K); 0]))*eye(size(K)); % increase numerical stability
    
    %% calculate function value and gradient
    alpha = K\y;
    
    L = 0.5*y'*alpha+0.5*sum(log(eig(K)))+log(2*pi); % - log ML
    
    dKdsigma_f = 2*(K-sigma_n^2*eye(2*n_obs))/sigma_f;
    dKdl       = sigma_f^2*[(rx.^2.*(l^2-ry.^2)-ry.^4+5*ry.^2*l^2-2*l^4)/l^7 ...
        rx.*ry.*(rx.^2+ry.^2-4*l^2)/l^7
        rx.*ry.*(rx.^2+ry.^2-4*l^2)/l^7 ...
        (ry.^2.*(l^2-rx.^2)-rx.^4+5*rx.^2*l^2-2*l^4)/l^7].*repmat(exp(-0.5*(rx.^2+ry.^2)/l^2),2,2)+sigma_n^2*eye(2*n_obs);
    dKdsigma_n = 2*sigma_n*eye(2*n_obs);
    
    beta = alpha*alpha'-K\eye(2*n_obs);
    
    g1 = -0.5*sum(sum(beta.*dKdsigma_f));
    g2 = -0.5*sum(sum(beta.*dKdl));
    g3 = -0.5*sum(sum(beta.*dKdsigma_n));
    g  = [g1 g2 g3]';
catch % case something goes wrong
    L=nan;
    g=nan(3,1);
end
end