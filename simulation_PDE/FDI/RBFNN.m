function S = RBFNN(X,center,eta)
%This function generate the function value provided by NN RBF at point X
%
%   X       Input vector, whose size must match the row number of 'center'
%   center	Coordinates of neural network centers
%   eta     Parameter used in Gaussian function

N = length(X);
S = exp(-ones(1,N)*((X-center).^2)/(2*eta)^2)';
