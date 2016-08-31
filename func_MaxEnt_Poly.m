% func_MaxEnt_Poly.m
% 
% Function called by Example_MaxEnt.m
% 
%-------------------------------------------------------------
%	xin =	nonlinear input sequence
%   N = order of the polynomials
%	x_lin = output (linearized) sequence
%   Ntrack_g = evolution of the objective function
%   NN = order of the used polynomial
%-------------------------------------------------------------
%
% J. Sole-Casals, K. Lopez-de-Ipiña, C. Caiafa
% August, 2016

function [x_lin,Ntrack_g,NN] = func_MaxEnt_Poly(xin,N)
if N>0 
    % using the predefined N
   [x_lin,Ntrack_g] = MaxEnt_Poly(xin,N);
   NN = N;
else  
    % looking for the quasi-optimal N
    % initializing variables...
    x_lin = xin;
    Ntrack_g = 0;
    col = 0;
    NN = N;
    for N = 1:15;
        col = col+1;
        [x1,track_g] = MaxEnt_Poly(xin,N);
        if mean(track_g)>mean(Ntrack_g)
            x_lin = x1;
            Ntrack_g = track_g;
            NN = N;
        end
    end
end

% other necessary functions...
function [xrec,track_g]=MaxEnt_Poly(x,N)
% variables and inicializations...
nu = 0.001;
iter = 0;
itmax= 500;
T = length(x);
e = x; % e is the nonlinear observation, before linearization process
a = 0.0*ones(N,1);
a(1) = 1; % initializing the polinomial to the identity function f(x)=x
b = zeros(N,length(x)); % initializing the matrix b to zero
for i = 0:N-1
    b(i+1,:) = (i+1)*(x.^i); 
end
% non-linear function output 
x = polyval(flipud([0; a]),e);
% iterations
track_g = [];
% main bucle
while iter < itmax
    iter = iter+1;
    % gradient 
    for i = 1:length(x)
        gradient(:,i) = b(:,i)/(a'*b(:,i));
        g(i) = log(abs(a'*b(:,i)));
    end
    gMax = mean(g); % function under maximization
    % weights update
    a = a+nu*mean(gradient,2);
	% tracking the gMax function
    track_g = [track_g gMax];
    % calculating the new x
    x = polyval(flipud([0; a]),e); 
    % normalization
    a = a/std(x);
    x = x-mean(x); 
    x = x/sqrt(mean(x.^2));
end
xrec = x;
