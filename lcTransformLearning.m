function [T, Z, W] = lcTransformLearning (X, labels, numOfAtoms, mu, lambda, eps)

% solves ||TX - Z||_Fro - mu*logdet(T) + eps*mu||T||_Fro + lambda||Q-WZ||_Fro

% Inputs
% X          - Training Data
% labels     - Class labels
% numOfAtoms - dimensionaity after Transform
% mu         - regularizer for Tranform
% lambda     - regularizer for coefficient
% eps        - regularizer for Transform
% type       - 'soft' or 'hard' update: default is 'soft'
% Output
% T          - learnt Transform
% Z          - learnt sparse coefficients
% W          - linear map
if nargin < 6
    eps = 1;
end
if nargin < 5
    lambda = 1;
end
if nargin < 4
    mu = 0.1;
end

maxIter = 10;
type = 'soft'; % default 'soft'

rng(1); % repeatable
T = randn(numOfAtoms, size(X,1));
Z = T*X;

numOfSamples = length(labels);
if min(labels) == 0
    labels = labels + 1;
end

numOfClass = max(labels);
Q = zeros(numOfClass,numOfSamples);
for i = 1:numOfSamples
    Q(labels(i),i) = 1;
end
W = Q / Z;

invL = (X*X' + mu*eps*eye(size(X,1)))^(-0.5);

for i = 1:maxIter 

    % update Transform T
    [U,S,V] = svd(invL*X*Z');
    D = [diag(diag(S) + (diag(S).^2 + 2*mu).^0.5) zeros(numOfAtoms, size(X,1)-numOfAtoms)];
    T = 0.5*V*D*U'*invL;
    
    % update Coefficients Z
    Z = (eye(size(W,2)) + lambda*W'*W)\(T*X + lambda*W'*Q);
    
    % update map
    W = Q / Z;
    
end
