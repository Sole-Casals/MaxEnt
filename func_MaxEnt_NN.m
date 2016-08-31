% func_MaxEnt_NN.m
% 
% Function called by Example_MaxEnt.m
% 
%-------------------------------------------------------------
%	xin =	nonlinear input sequence
%   mu = adaptation step. Experimental good values are about 10 for 
%    tanh(·) type distortions, and about 0.1 for x(·)^3 distortions
%	x_lin = output (linearized) sequence
%   track_g = evolution of the objective function
%-------------------------------------------------------------
%
% J. Sole-Casals, K. Lopez-de-piña, C. Caiafa
% August, 2016

function [x_lin,track_g] = func_MaxEnt_NN(xin, mu)
% variables initialization
T = length(xin); % number of points (signal length)
e = xin; % observed signal, to be linearized
itmax = 15000; % maximum number of iterations
iter = 1; 
track_g(1) = -Inf; % to track the function under maximization
n = 7; % number of neurons in the hidden layer

% neural network (NN) initialization
[a,b,c] = initialization(xin,n);

% signal at the output of the NN
for t = 1:T
 xin(t) = xn(c',b',a',e(t));
end;

% main bucle
while iter < itmax
   for t=1:length(e)
      theta(:,1) = sigmaP(c',b',e(t)); % c and b have to be row vectors
      fi(:,1) = sigmaPP(c',b',e(t)); % c and b have to be row vectors
      deno(t) = (a.*c)'*theta; % denominator used in the next lines
      % gradients
      grad_a(:,t) = (c.*theta)/deno(t);
      grad_b(:,t) = -(a.*c.*fi)/deno(t);
      grad_c(:,t) = (a.*theta+a.*c.*fi*e(t))/deno(t);
      % objective function 
      g(t) = log(deno(t));
   end;
   % weights update
   a = a+mu*mean(grad_a,2);
   b = b+mu*mean(grad_b,2);
   c = c+mu*mean(grad_c,2);
   gMax = mean(g); % function under maximization
   track_g = [track_g gMax]; % tracking the gMax function
  	% calculating the new xin
	for t = 1:length(e)
        xin(t) = xn(c',b',a',e(t));
	end;
    % normalization
    a = a/std(xin); 
    xin = xin-mean(xin); 
    xin = xin/std(xin); 
    % checking stop condition
    if ((iter>1000) && (abs(track_g(end)-track_g(end-1))<1e-6)) || (iter>=itmax)
        break;
    else
        iter = iter+1;
    end
end;
x_lin = xin;

%%%% other additional functions
% xn
function Y = xn(c,b,a,e)
% c: row vector, weight of the input to each unit
% b: row vector, bias
% a: row vector, weight of the units to the output
% e: scalar input 
Y = a*ssigma(c*e-b)';
% inicialization
function [a,b,c] = initialization(x,N)
xmin = min(x); 
xmax= max(x);
step = (xmax-xmin)/N;
a = ones(N,1);
for n = 1:N
    b(n,1) = step*(2*n-1)/2;
end
b = b-(xmax-xmin)/2;
param = 1/(2*xmax);
c = param*ones(N,1);
% ssigma
function Y = ssigma(x)
% Y output vector
% x input vector
Y = 1./(1+exp(-x)); % logistic function
% dsigma
function Y = dsigma(x)
% Y output vector
% x input vector
Y = exp(-x)./(1+exp(-x)).^2;	% first derivative of the logistic function
% ddsigma
function Y = ddsigma(x)
% Y output vector
% x input vector
Y = exp(-x).*(-1+exp(-x))./(1+exp(-x)).^3; % second derivative of the logistic function
% sigmaP
function Y = sigmaP(c,b,e)
% c: row vector, weight of the input to each unit
% b: row vector, bias
% e: scalar input 
Y = dsigma(c*e-b);
% sigmaPP
function Y = sigmaPP(c,b,e)
% c: row vector, weight of the input to each unit
% b: row vector, bias
% e: scalar input 
Y=ddsigma(c*e-b);
