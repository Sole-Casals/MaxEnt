% Maximization of Entropy
% 
% This is the main file of the algorithm presented in the paper
% "Inverting Monotonic Nonlinearities by Entropy Maximization".
% You can play with it by generating some PNL mixtures and invert the
% nonlinear distortion through MaxEnt, Gaussianization and Uniformization
% algorithms.
% 
% How it works:
% 1) choose a nonlinear distortion by uncommenting lines 69-71 or 73-75
% 2) choose an algorithm by uncommenting lines 79-80 or 82-83
%  
% You can play with the order of the polynomes (N), the length of the
% signals (T), the matrix (A), etc. Other parameters can be tunned into the
% corresponding functions.
% 
% Main variables:
%  N: order of the polynomes, only used when polynomial parametrization is
%   selected. If N=0, the algorithm automatically looks for the quasi-optimal 
%   N from 1 to 15.
%  T: length of the signals
%  n: number of sources. You can increase it, but in this demo only the
%   first two channels are processed.
%  s: source signals
%  x: linear observations
%  A: mixing matrix
%  e: non-linear observations
%  mu: adaptation step for the NN parameterization. Experimental good values are 
%   about 10 for tanh type distortions, and about 0.1 for x^3 distortions
%  z: linear estimated mixtures using MaxEnt algorithm
%  zg: linear estimated mixtures using Gaussianization algorithm
%  zu: linear estimated mixtures using Uniformization algorithm
% 
% J. Sole-Casals, K. López-de-Ipiña, C. Caiafa
% August, 2016

%%
clc
clear all;
close all;

% parameters
N =  10; % order of the polynomials (0 for automatic searching)  
T = 500; % length of the signals
n = 2; % number of sources

% mixing matrix
rng('shuffle')
A = rand(n,n);
A(logical(eye(size(A)))) = 1;

% source signals
s = [];
for c = 1:n
    rng('shuffle')
    s1 = ((rand([1 T])-0.5));
    % normalization
    s1 = s1-mean(s1);
    s1 = s1/std(s1);
    s = [s; s1];
end
        
% mixed signals
x = A*s;
e = x;

% distorted signals 
%%% tanh type
e(1,:) = tanh(3*x(1,:))+0.1*x(1,:);
e(2,:) = tanh(3*x(2,:))+0.1*x(2,:); 
mu = 10;
%%% x^3 type
% e(1,:) = x(1,:).^3+0.1*x(1,:);
% e(2,:) = x(2,:).^3+0.1*x(2,:); 
% mu = 0.1;

% MaxEnt, using Poly or NN
%%% Poly
[z1,track_g1,N1] = func_MaxEnt_Poly(e(1,:)',N);
[z2,track_g2,N2] = func_MaxEnt_Poly(e(2,:)',N);
%%% NN
% [z1,track_g1] = func_MaxEnt_NN(e(1,:)',mu);
% [z2,track_g2] = func_MaxEnt_NN(e(2,:)',mu);

% linearized observations
z = [z1'; z2'];

% Gaussianization and Uniformization  
[zg1, zu1] = func_GaussUnif(e(1,:));
[zg2, zu2] = func_GaussUnif(e(2,:));
zg = [zg1; zg2];
zu = [zu1; zu2];

% scale factor (same energy), to compare performance
z(1,:) = z(1,:)*norm(x(1,:))/norm(z(1,:));
z(2,:) = z(2,:)*norm(x(2,:))/norm(z(2,:)); 
zg(1,:) = zg(1,:)*norm(x(1,:))/norm(zg(1,:));
zg(2,:) = zg(2,:)*norm(x(2,:))/norm(zg(2,:));
zu(1,:) = zu(1,:)*norm(x(1,:))/norm(zu(1,:));
zu(2,:) = zu(2,:)*norm(x(2,:))/norm(zu(2,:));

% SNR for each channel and algorithm
snr1z = func_snr(z(1,:),x(1,:));
snr2z = func_snr(z(2,:),x(2,:)); 
snr1g = func_snr(zg(1,:),x(1,:));
snr2g = func_snr(zg(2,:),x(2,:));
snr1u = func_snr(zu(1,:),x(1,:));
snr2u = func_snr(zu(2,:),x(2,:));

% showing results on the screen
disp(['SNR for channel 1 using MaxEnt:          ',num2str(snr1z),' dB'])
disp(['SNR for channel 2 using MaxEnt:          ',num2str(snr2z),' dB'])
disp(['SNR for channel 1 using Gaussianization: ',num2str(snr1g),' dB'])
disp(['SNR for channel 2 using Gaussianization: ',num2str(snr2g),' dB'])
disp(['SNR for channel 1 using Uniformization:  ',num2str(snr1u),' dB'])
disp(['SNR for channel 2 using Uniformization:  ',num2str(snr2u),' dB'])
