% func_GaussUnif.m
%
% Function called by Example_MaxEnt.m
%
% We estimate the sample of the signal zg, obtained from the signal e in order to zg have a unit power gaussian distribution
% We estimate the sample of the signal zu, obtained from the signal e in order to zu have a unit power uniform distribution
%
% J. Sole-Casals, K. Lopez-de-piña, C. Caiafa
% August, 2016

function [zg,zu] = func_GaussUnif(e)
N = length(e);
[eo,ei] = sort(e);
Fe = zeros(1,N);
for t = 1:N
   Fe(t) = (t-0.5)/N;
end
for t = 1:N
   in = find(ei==t);
   zg(t) = norminv(Fe(in));
   zu(t) = Fe(in)-0.5; % equivalent to unifinv(Fe(in),min(e),max(e));
end



