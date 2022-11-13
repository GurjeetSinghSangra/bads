function [y, noise] = ackley_he_noisy(x)
f = -20 * exp(-0.2 * sqrt( sum(x .* x)/ numel(x))) ...
        - exp(sum(cos(2*pi*x)) / numel(x)) ...
        + 20 + 2.7182818284590452353602874713526625;

noise = lognrnd(0,1) + 1 + 0.1 .* (f - zeros(size(x, 1)));
y = f + noise;
end