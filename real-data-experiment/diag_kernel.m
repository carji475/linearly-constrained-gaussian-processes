function [K,varargout] = diag_kernel(theta,varargin)
% Returns the Gram matrix and either of 
%   i) the partial derivative matrices (nargin=4)
%   ii) the cross covariance for prediction (nargin=3)
%
% INPUTS
%   theta - kernel parameters (signal variance, length scale, linear mult factor, noise std)
%   varargin{1} - input positions of measurements (3 columns) 
%   varargin{2} - input positions of artificial curl measurements (3 columns)
%   if nargin=4
%         varargin{3} - input positions of predictions (3 columns)
%
% OUTPUTS
%   K - Gram matrix
%   if nargin=4
%       varargout - cross covariance for prediction
%   if nargin=3
%       varargout - tensor with partial derivative matrices of the Gram
%       matrix
%

%% extract data
Xfield = varargin{1};   % magnetic field input positions
Xcurl  = varargin{2};   % curl observation positions

% hyperparameters
sigmaf = exp(theta(1)/2); l = exp(theta(2)/2);
sigmab = exp(theta(3)/2); sigman = exp(theta(4)/2);

%% inputs
nr_field = size(Xfield,1);
nr_curl = size(Xcurl,1);

%% construct some helping matrices
% field/field
rx_field = repmat(Xfield(:,1),1,nr_field)-repmat(Xfield(:,1)',nr_field,1);
ry_field = repmat(Xfield(:,2),1,nr_field)-repmat(Xfield(:,2)',nr_field,1);
rz_field = repmat(Xfield(:,3),1,nr_field)-repmat(Xfield(:,3)',nr_field,1);

% field/curl
rx_field_curl = repmat(Xfield(:,1)',nr_curl,1)-repmat(Xcurl(:,1),1,nr_field);
ry_field_curl = repmat(Xfield(:,2)',nr_curl,1)-repmat(Xcurl(:,2),1,nr_field);
rz_field_curl = repmat(Xfield(:,3)',nr_curl,1)-repmat(Xcurl(:,3),1,nr_field);

% curl/curl
rx_curl = repmat(Xcurl(:,1),1,nr_curl)-repmat(Xcurl(:,1)',nr_curl,1);
ry_curl = repmat(Xcurl(:,2),1,nr_curl)-repmat(Xcurl(:,2)',nr_curl,1);
rz_curl = repmat(Xcurl(:,3),1,nr_curl)-repmat(Xcurl(:,3)',nr_curl,1);

% exponential forms
exp_field       = exp(-0.5*(rx_field.^2+ry_field.^2+rz_field.^2)/l^2);
exp_field_curl  = exp(-0.5*(rx_field_curl.^2+ry_field_curl.^2+rz_field_curl.^2)/l^2);
exp_curl        = exp(-0.5*(rx_curl.^2+ry_curl.^2+rz_curl.^2)/l^2);

%% Gram matrix blocks
% field/field
K_field = [sigmaf^2*exp_field+sigmab^2*ones(nr_field) zeros(nr_field,2*nr_field)
    zeros(nr_field,nr_field) sigmaf^2*exp_field+sigmab^2*ones(nr_field) zeros(nr_field,nr_field)
    zeros(nr_field,2*nr_field) sigmaf^2*exp_field+sigmab^2*ones(nr_field)];
% add noise
K_field = K_field + sigman^2*eye(size(K_field));

% field/curl
K_field_curl = [zeros(nr_curl,nr_field) ...
    (1/l^2)*rz_field_curl*sigmaf^2.*exp_field_curl ...
    -(1/l^2)*ry_field_curl*sigmaf^2.*exp_field_curl
    
    -(1/l^2)*rz_field_curl*sigmaf^2.*exp_field_curl ...
    zeros(nr_curl,nr_field) ...
    (1/l^2)*rx_field_curl*sigmaf^2.*exp_field_curl
    
    (1/l^2)*ry_field_curl*sigmaf^2.*exp_field_curl ...
    -(1/l^2)*rx_field_curl*sigmaf^2.*exp_field_curl ...
    zeros(nr_curl,nr_field)];

% curl/curl
K_curl = [(1/l^2)*(1-(1/l^2)*rz_curl.^2)*sigmaf^2.*exp_curl+...
    (1/l^2)*(1-(1/l^2)*ry_curl.^2)*sigmaf^2.*exp_curl ...
    (1/l^4)*rx_curl.*ry_curl*sigmaf^2.*exp_curl ...
    (1/l^4)*rx_curl.*rz_curl*sigmaf^2.*exp_curl
    
    (1/l^4)*rx_curl.*ry_curl*sigmaf^2.*exp_curl ...
    (1/l^2)*(1-(1/l^2)*rz_curl.^2)*sigmaf^2.*exp_curl+...
    (1/l^2)*(1-(1/l^2)*rx_curl.^2)*sigmaf^2.*exp_curl ...
    (1/l^4)*rz_curl.*ry_curl*sigmaf^2.*exp_curl
    
    (1/l^4)*rx_curl.*rz_curl*sigmaf^2.*exp_curl ...
    (1/l^4)*rz_curl.*ry_curl*sigmaf^2.*exp_curl ...
    (1/l^2)*(1-(1/l^2)*ry_curl.^2)*sigmaf^2.*exp_curl+...
    (1/l^2)*(1-(1/l^2)*rx_curl.^2)*sigmaf^2.*exp_curl];

% complete Gram matrix
K = [K_field K_field_curl'; K_field_curl K_curl];

%% prediction covariance
if nargin==4
    Xpred=varargin{3};
    nr_pred = size(Xpred,1);
    % field/pred
    rx_field_pred = repmat(Xfield(:,1)',nr_pred,1)-repmat(Xpred(:,1),1,nr_field);
    ry_field_pred = repmat(Xfield(:,2)',nr_pred,1)-repmat(Xpred(:,2),1,nr_field);
    rz_field_pred = repmat(Xfield(:,3)',nr_pred,1)-repmat(Xpred(:,3),1,nr_field);
    % curl/pred
    rx_curl_pred = repmat(Xcurl(:,1)',nr_pred,1)-repmat(Xpred(:,1),1,nr_curl);
    ry_curl_pred = repmat(Xcurl(:,2)',nr_pred,1)-repmat(Xpred(:,2),1,nr_curl);
    rz_curl_pred = repmat(Xcurl(:,3)',nr_pred,1)-repmat(Xpred(:,3),1,nr_curl);
    exp_field_pred = exp(-0.5*(rx_field_pred.^2+ry_field_pred.^2+rz_field_pred.^2)/l^2);
    exp_curl_pred = exp(-0.5*(rx_curl_pred.^2+ry_curl_pred.^2+rz_curl_pred.^2)/l^2);
    % field/pred
    k_field_pred = blkdiag(sigmaf^2*exp_field_pred+sigmab^2*ones(nr_pred,nr_field),...
        sigmaf^2*exp_field_pred+sigmab^2*ones(nr_pred,nr_field),...
        sigmaf^2*exp_field_pred+sigmab^2*ones(nr_pred,nr_field));
    % curl/pred
    k_curl_pred = ...
        [zeros(nr_pred,nr_curl) ...
        (1/l^2)*rz_curl_pred*sigmaf^2.*exp_curl_pred ...
        -(1/l^2)*ry_curl_pred*sigmaf^2.*exp_curl_pred
        -(1/l^2)*rz_curl_pred*sigmaf^2.*exp_curl_pred ...
        zeros(nr_pred,nr_curl) ...
        (1/l^2)*rx_curl_pred*sigmaf^2.*exp_curl_pred
        (1/l^2)*ry_curl_pred*sigmaf^2.*exp_curl_pred ...
        -(1/l^2)*rx_curl_pred*sigmaf^2.*exp_curl_pred ...
        zeros(nr_pred,nr_curl)];
    varargout{1} = [k_field_pred k_curl_pred];
end

%% derivatives
if nargin==3
    normQuad_field=(rx_field.^2+ry_field.^2+rz_field.^2);
    % field/field - sigmaf
    dK_field_dsigmaf = [sigmaf^2*exp_field zeros(nr_field,2*nr_field)
        zeros(nr_field,nr_field) sigmaf^2*exp_field zeros(nr_field,nr_field)
        zeros(nr_field,2*nr_field) sigmaf^2*exp_field];
    % field/field - l
    dK_field_dl = (1/2)*[sigmaf^2*(1/l^2)*normQuad_field.*exp_field zeros(nr_field,2*nr_field)
        zeros(nr_field) sigmaf^2*(1/l^2)*normQuad_field.*exp_field zeros(nr_field)
        zeros(nr_field,2*nr_field) sigmaf^2*(1/l^2)*normQuad_field.*exp_field];
    % field/field - sigmab
    dK_field_dsigmab = [sigmab^2*ones(nr_field) zeros(nr_field,2*nr_field)
        zeros(nr_field,nr_field) sigmab^2*ones(nr_field) zeros(nr_field,nr_field)
        zeros(nr_field,2*nr_field) sigmab^2*ones(nr_field)];
    % field/field - sigma_n
    dK_field_dsigman= sigman^2*eye(3*nr_field);
    
    normQuad_field_curl = (rx_field_curl.^2+ry_field_curl.^2+rz_field_curl.^2); % quadratic norm
    % field/curl - sigmaf
    dK_field_curl_dsigmaf = [zeros(nr_curl,nr_field) ...
        (1/l^2)*rz_field_curl*sigmaf^2.*exp_field_curl ...
        -(1/l^2)*ry_field_curl*sigmaf^2.*exp_field_curl
        -(1/l^2)*rz_field_curl*sigmaf^2.*exp_field_curl ...
        zeros(nr_curl,nr_field) ...
        (1/l^2)*rx_field_curl*sigmaf^2.*exp_field_curl
        (1/l^2)*ry_field_curl*sigmaf^2.*exp_field_curl ...
        -(1/l^2)*rx_field_curl*sigmaf^2.*exp_field_curl ...
        zeros(nr_curl,nr_field)];
    % field/curl - l
    dK_field_curl_dl = (1/2)*[zeros(nr_curl,nr_field) ...
        -( 2-(1/l^2)*normQuad_field_curl )*(1/l^2).*rz_field_curl*sigmaf^2.*exp_field_curl ...
        ( 2-(1/l^2)*normQuad_field_curl )*(1/l^2).*ry_field_curl*sigmaf^2.*exp_field_curl
        ( 2-(1/l^2)*normQuad_field_curl )*(1/l^2).*rz_field_curl*sigmaf^2.*exp_field_curl ...
        zeros(nr_curl,nr_field) ...
        -( 2-(1/l^2)*normQuad_field_curl )*(1/l^2).*rx_field_curl*sigmaf^2.*exp_field_curl
        -( 2-(1/l^2)*normQuad_field_curl )*(1/l^2).*ry_field_curl*sigmaf^2.*exp_field_curl ...
        ( 2-(1/l^2)*normQuad_field_curl )*(1/l^2).*rx_field_curl*sigmaf^2.*exp_field_curl ...
        zeros(nr_curl,nr_field)];
    
    normQuad_curl = (rx_curl.^2+ry_curl.^2+rz_curl.^2); % quadratic norm
    % curl/curl - sigmaf
    dK_curl_dsigmaf = [zeros(nr_curl,3*nr_curl)
        zeros(nr_curl,nr_curl) ...
        (1/l^2)*(1-(1/l^2)*rz_curl.^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)...
        (1/l^4)*rz_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)
        zeros(nr_curl) ...
        (1/l^4)*rz_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        (1/l^2)*(1-(1/l^2)*ry_curl.^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)]+...
        [(1/l^2)*(1-(1/l^2)*rz_curl.^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        zeros(nr_curl) ...
        (1/l^4)*rx_curl.*rz_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)
        zeros(nr_curl,3*nr_curl)
        (1/l^4)*rx_curl.*rz_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        zeros(nr_curl) ...
        (1/l^2)*(1-(1/l^2)*rx_curl.^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)]+...
        [(1/l^2)*(1-(1/l^2)*ry_curl.^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        (1/l^4)*rx_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        zeros(nr_curl)
        (1/l^4)*rx_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        (1/l^2)*(1-(1/l^2)*rx_curl.^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        zeros(nr_curl)
        zeros(nr_curl,3*nr_curl)];
    
    dK_curl_dl = 0.5*( [zeros(nr_curl,3*nr_curl)
        zeros(nr_curl,nr_curl) ...
        ( -2+(1/l^2)*( normQuad_curl+ 4*rz_curl.^2 ) - (1/l^4)*rz_curl.^2.*normQuad_curl ).*(1/l^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)...
        -(1/l^4)*rz_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2).*(4-(1/l^2)*normQuad_curl)
        zeros(nr_curl) ...
        -(1/l^4)*rz_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2).*(4-(1/l^2)*normQuad_curl) ...
        ( -2+(1/l^2)*( normQuad_curl+ 4*ry_curl.^2 ) - (1/l^4)*ry_curl.^2.*normQuad_curl ).*(1/l^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)]+...
        [( -2+(1/l^2)*( normQuad_curl+ 4*rz_curl.^2 ) - (1/l^4)*rz_curl.^2.*normQuad_curl ).*(1/l^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        zeros(nr_curl) ...
        -(1/l^4)*rx_curl.*rz_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2).*(4-(1/l^2)*normQuad_curl)
        zeros(nr_curl,3*nr_curl)
        -(1/l^4)*rx_curl.*rz_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2).*(4-(1/l^2)*normQuad_curl) ...
        zeros(nr_curl) ...
        ( -2+(1/l^2)*( normQuad_curl+ 4*rx_curl.^2 ) - (1/l^4)*rx_curl.^2.*normQuad_curl ).*(1/l^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2)]+...
        [( -2+(1/l^2)*( normQuad_curl+ 4*ry_curl.^2 ) - (1/l^4)*ry_curl.^2.*normQuad_curl ).*(1/l^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        -(1/l^4)*rx_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2).*(4-(1/l^2)*normQuad_curl) ...
        zeros(nr_curl)
        -(1/l^4)*rx_curl.*ry_curl*sigmaf^2.*exp(-0.5*normQuad_curl/l^2).*(4-(1/l^2)*normQuad_curl) ...
        ( -2+(1/l^2)*( normQuad_curl+ 4*rx_curl.^2 ) - (1/l^4)*rx_curl.^2.*normQuad_curl ).*(1/l^2)*sigmaf^2.*exp(-0.5*normQuad_curl/l^2) ...
        zeros(nr_curl)
        zeros(nr_curl,3*nr_curl)] );
    
    % total
    dKd(:,:,1) = [dK_field_dsigmaf dK_field_curl_dsigmaf'; dK_field_curl_dsigmaf dK_curl_dsigmaf];
    dKd(:,:,2) = [dK_field_dl dK_field_curl_dl'; dK_field_curl_dl dK_curl_dl];
    dKd(:,:,3) = [dK_field_dsigmab zeros(3*nr_field,3*nr_curl); zeros(3*nr_curl,3*nr_field+3*nr_curl)];
    dKd(:,:,4) = [dK_field_dsigman zeros(3*nr_field,3*nr_curl); zeros(3*nr_curl,3*nr_field+3*nr_curl)];
    
    varargout{1} = dKd;
end
end