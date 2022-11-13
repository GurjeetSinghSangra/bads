%% Example 7: Noisy Ackley function

% We test BADS on a noisy function. For the purpose of this test, we
% manually add unit Gaussian noise to the 'sphere' (quadratic) function.

% Define the noisy objective as a function handle


%ackleyfunc = @(x) -20 * exp(-0.2 * sqrt( sum(x .* x)/ numel(x))) ...
        %- exp(sum(cos(2*pi*x)) / numel(x)) ...
        %+ 20 + 2.7182818284590452353602874713526625;
noisyfun = @ackley_he_noisy;


% This time to help the search we set tighter bounds

lb = [-32 -32];   ub = [32 32];
plb = [-32 -32];  pub = [32 32];
load('./X0_examples/runs_20/x0_Noisy Ackley.mat');

res = zeros(20, 4);
seed = 3;
options.UncertaintyHandling = 1;
options.SpecifyTargetNoise = 'yes';
for i = 1:size(X0, 1)
    rng(seed, 'twister')
    x0 = X0(1, :);
    [x,fval,exitflag,output] = bads(noisyfun,x0,lb,ub,plb,pub, [], options);
    res(i, 1) = fval;
    res(i, [2:3]) = x;
    res(i, 4) = output.funccount;
    seed = seed + 1;
end

histfit(res(:, 1), 30)
title(sprintf('BADS Fvals hist n f-evals: %0.3f', mean(res(:, 4))))
xlabel('fvals')
saveas(gcf,'hist_fvals.png')

pts = linspace(-3, 3, length(res));
scatter(res(:, 2), res(:, 3), 25, pts, 'filled', 'magenta');
title('BADS Scatter plot final X')
xlabel('x1')
xlabel('x2')
saveas(gcf,'scatter_X.png')

xgrid=linspace(-0.5,0.5);
ygrid=linspace(-0.5,0.5);
[x1,y1] = meshgrid(xgrid, ygrid);
xi = [x1(:) y1(:)];
[f,ep]=ksdensity([res(:, 2) res(:, 3)], xi );
X = reshape(ep(:,1),length(xgrid),length(ygrid));
Y = reshape(ep(:,2),length(xgrid),length(ygrid));
Z = reshape(f,length(xgrid),length(ygrid));
contourf(X,Y,Z,10)
ax=gca;
ax.XAxis.Exponent = 0;
xtickformat('%.4f')
xlabel('col1')
ylabel('col2')
colorbar
%colormap(res(:, [2, 3]), 15)
% Run BADS on the noisy function


% FVAL in this case is an *estimate* of the function at X, obtained by
% averaging ten function evaluations. These values can be found in the
% OUTPUT structure, together with additional information about the optimization.

display('The returned OUTPUT structure is:');
output
