
function [PC, mn, V] = pca(data)
% PCA: Perform PCA using SVD.
% data - MxN matrix of input data
% (M dimensions, N trials)
% dim - projection dimension
% projected - Mxdim matrix of projected data
% recon - MxN matrix of recontruction form the projected data
% PC - each column is a PC
% V - Mx1 matrix of variances

[M,N] = size(data);

% subtract off the mean for each dimension
mn = mean(data,2);
data = data - repmat(mn,1,N);

% construct the matrix Y
Y = data' / sqrt(N-1);

% SVD does it all
[u,S,PC] = svd(Y);

% calculate the variances
S = diag(S);
V = S .* S;

% % project the original data
% projected = PC(:,1:dim)' * data;
% 
% % reconstruct the projected data to the original data space
% recon = PC(:,1:dim) * projected + repmat(mn,1,N);

end