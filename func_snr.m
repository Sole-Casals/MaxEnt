% func_snr(x,y)
%
% Function called by Example_MaxEnt.m
%
% We estimate the SNR of signals x and y
%
% J. Sole-Casals, K. Lopez-de-piña, C. Caiafa
% August, 2016

function snr = func_snr(x,y)

lx = length(x);
[temp,i] = max(xcorr(x,y)); 
j = abs(lx-i)+1;
yy = y(j:lx);
xx = x(1:lx-j+1);
snr = 10*log10(sum(xx.^2)/sum((xx-yy).^2));
