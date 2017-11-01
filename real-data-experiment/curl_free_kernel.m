function [K,dKd]=curl_free_kernel(theta,X1,X2)
    % [K,dKdl,dKdb] = curl_free_kernel(X1,X2,theta)    
    % Compute the covariance matrix for a curl-free GP        
    % X1 - First set of input data
    % X2 - Second set of input data
    % theta - The hyperparameters
    % theta = [0.3,  theta, -0.4]
    sigma_f=exp(theta(1)/2); %The standard diviation of the gaussian process
    l=exp(theta(2)/2); %Characteristic length scale of the H-field
    sigma_b=exp(theta(3)/2); %Standard diviation of earth magnetic field components            
    
    % Find the size of the data
    m=size(X1,2);
    n1=size(X1,1);
    n2=size(X2,1);    
        
    % Construct a matrix [n1,n2,m]=size(Xdiff) where
    % Xdiff(i,j,:)=X1(:,i)-X2(:,j)
    Xdiff=repmat(permute(X1',[2 3 1]),[1 n2 1])-repmat(permute(X2',[3 2 1]),[n1 1 1]);
    
    % Construct a matrix [n1,n2]=size(Xdiff_sum) where
    % Xdiff_square(i,j)=norm(X1(:,i)-X2(:,j))^2
    Xdiff_square=sum(Xdiff.^2,3);
    
    % Construct a matrix [n1,n2]=size(Xdiff_sum) where
    % Xdiff_square_norm(i,j)=norm(X1(:,i)-X2(:,j))^2/l^2
    Xdiff_square_norm=Xdiff_square/l^2;
    
    % Construct a tensor [n1, n2, m, m]=size(Kgain), 
    % where Kgain(i,j,:,:)=exp(-norm(X1(:,i)-X2(:,j))^2/(2*sigma^2))*ones(m,m)     
    Kgain=repmat(sigma_f^2*exp(-Xdiff_square_norm/2),[1 1 m m]);
    % alpha=1/2;
    % Kgain=repmat(sigma_f^2*(1+Xdiff_square/(alpha*2)).^(-alpha),[1 1 m m]);
    % Kgain=repmat(sigma_f^2*(1+sqrt(3*Xdiff_square)).*exp(-sqrt(3*Xdiff_square)),[1 1 m m]);       
    
    % Contruct a matrix [n1,n2,m,m]=size(Xprod), where
    % Xprod_temp(i,j,:,:)=(X1(:,i)-X2(:,j))*(X1(:,i)-X2(:,j))/(sigma^2)
    Xdiff1=repmat(Xdiff/l,[1 1 1 m]);
    Xdiff2=permute(Xdiff1,[1 2 4 3]);
    Xprod=Xdiff1.*Xdiff2;
    
    % Contruct a tensor [n1,n2,m,m]=size(eyeBig), where
    % eyeBig(i,j,:,:)= eye(m)
    eyeBig=repmat(permute(eye(m),[3 4 1 2]),[n1 n2 1 1]);
                   
        
    % Contruct a tensor [n1,n2,m,m]=size(K_temp), where
    % K_temp(i,j,:,:) is the curl free kernel evaluated at X1(:,i)
    % and X2(:,j)
    K_curl_free=Kgain.*(eyeBig-Xprod);        
    
       
    % Contruct a tensor [n1,n2,m,m]=size(priorBig), where
    % priorBig(i,j,:,:) is the sigma_b^2*eye(m)
    % K_M=Kgain.*(eyeBig+eyeBig_scaled);
    B = sigma_b^2*eye(m);    
    priorBig=repmat(permute(B,[3 4 1 2]),[n1 n2 1 1]);      
    
    % Build up the augmentat covariance matrix    
    K_temp=K_curl_free+priorBig;            
            
    % Construct a matrix [n1*m, n2*m]=size(K) by ungen the chartesian coordinates first
    % permute_vec = [3 1 4 2];
    permute_vec = [1 3 2 4];
    K=reshape(permute(K_temp,permute_vec),[n1*m n2*m]);
    
    % Add noise
    if length(theta) > 3 
        try
            sigma_n=exp(theta(4)/2); % Standard diviation of earth magnetic field components
            noise_eye = sigma_n^2*eye(size(K,1));
            K = K + noise_eye;
        catch
        end
    end
        
    if nargout > 1        
        % Compute dKdb       
        dKd = zeros([size(K) length(theta)]);                
        dKd(:,:,1)=reshape(permute(K_curl_free,permute_vec),[n1*m n2*m]);
                        
        % Compute dKdl        
        dKd(:,:,2) = reshape(permute(Kgain.*Xprod + K_curl_free.*repmat(Xdiff_square_norm/2,[1 1 m m]),permute_vec),[n1*m n2*m]);
        
        % Compute dKdb        
        dKd(:,:,3)=reshape(permute(priorBig,permute_vec),[n1*m n2*m]);
        
        % Add noise
        if length(theta) > 3 
            dKd(:,:,4)= noise_eye;
        end
    end 
end