function [y, noise] = quadnoisy(x)
noise = lognrnd(0,1) + sqrt(abs(min(x)));
y = sum(x.^2 -8.*x + 16, 2) + noise;
end