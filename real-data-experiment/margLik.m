function [L,grad] = margLik(theta,Xfield,Xcurl,yfield,kerneltype)
% negative log marginal likelihood and gradient    

    if strcmpi(kerneltype,'diag')
        [K,dKd] = diag_kernel(theta,Xfield,Xcurl);
        y=[yfield; zeros(3*size(Xcurl,1),1)];
    else
        [K,dKd] = curl_free_kernel(theta,Xfield,Xfield);
        y=yfield;
    end
    K = K+2*abs(min([eig(K); 0]))*eye(size(K)); % add some numerical regularisation

    alpha = K\y;
    L     = 0.5*y'*alpha+0.5*sum(log(eig(K))); % - log ML
    
    beta  = alpha*alpha'-K\eye(size(K));
    
    grad1 = -0.5*sum(sum(beta.*dKd(:,:,1)));
    grad2 = -0.5*sum(sum(beta.*dKd(:,:,2)));
    grad3 = -0.5*sum(sum(beta.*dKd(:,:,3)));
    grad4 = -0.5*sum(sum(beta.*dKd(:,:,4)));
    grad=[grad1 grad2 grad3 grad4]'; % gradient
end