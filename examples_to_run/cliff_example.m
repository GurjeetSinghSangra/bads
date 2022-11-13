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


%% Example 7: Noisy Cliff function

% We test BADS on a noisy function. For the purpose of this test, we
% manually add unit Gaussian noise to the 'sphere' (quadratic) function.

% Define the noisy objective as a function handle


noisyfun = @(x) sum(x.*x) + 1e4*sum(x < 0) +  randn(size(x,1),1);

lb = [-20 -20];   ub = [20 20];
plb = [-20 -20];  pub = [20 20];
load('./X0_examples/runs_20/x0_Noisy Cliff.mat');

res = zeros(20, 4);
seed = 3;
hold on 
h = zeros(size(X0, 1), 1);
for i = 1:size(X0, 1)
    rng(seed, 'twister')
    x0 = X0(1, :);
    [x,fval,exitflag,output, optimState] = bads(noisyfun,x0,lb,ub,plb,pub);
    h(i) = plot(optimState.iterList.funccount, optimState.iterList.fval);
    res(i, 1) = fval;
    res(i, [2:3]) = x;
    res(i, 4) = output.funccount;
    seed = seed + 1;
end
hold off
legend(h, "run " + [1:20]);
ylabel('fval')
xlabel('n f-evals')
saveas(gcf,'history_trajectories_cliff.png')

histfit(res(:, 1), 30)
title(sprintf('BADS Fvals hist n f-evals: %0.3f', mean(res(:, 4))))
xlabel('fvals')
saveas(gcf,'history_fvals_cliff.png')

pts = linspace(-3, 3, length(res));
scatter(res(:, 2), res(:, 3), 25, pts, 'filled', 'magenta');
title('BADS Scatter plot final X')
xlabel('x1')
xlabel('x2')
saveas(gcf,'scatter_X_cliff.png')


%% Example Rastrigin

noisyfun = @(x) sum(x .^ 2 - 10 * cos(2*pi*x) + 10) +  randn(size(x,1),1);

lb = [-20 -20];   ub = [20 20];
plb = [-5.12 -5.12];  pub = [5.12 5.12];
load('./x0_Noisy Rastrigin.mat');

res = zeros(20, 4);
seed = 3;
hold on 
h = zeros(size(X0, 1), 1);
for i = 1:size(X0, 1)
    rng(seed, 'twister')
    x0 = X0(1, :);
    [x,fval,exitflag,output, optimState] = bads(noisyfun,x0,lb,ub,plb,pub);
    h(i) = plot(optimState.iterList.funccount, optimState.iterList.fval);
    res(i, 1) = fval;
    res(i, [2:3]) = x;
    res(i, 4) = output.funccount;
    seed = seed + 1;
end
hold off
legend(h, "run " + [1:20]);
ylabel('fval')
xlabel('n f-evals')
saveas(gcf,'history_trajectories_rastrigin.png')

histfit(res(:, 1), 30)
title(sprintf('BADS Fvals hist n f-evals: %0.3f', mean(res(:, 4))))
xlabel('fvals')
saveas(gcf,'history_fvals_rastrigin.png')

pts = linspace(-3, 3, length(res));
scatter(res(:, 2), res(:, 3), 25, pts, 'filled', 'magenta');
title('BADS Scatter plot final X')
xlabel('x1')
xlabel('x2')
saveas(gcf,'scatter_X_rastrigin.png')

%% Multiple objective runs
%Noisy Cliff
cliff = @(x) sum(x.*x) + 1e4*sum(x < 0);
% Rastrigin
rastrigin = @(x) sum(x .^ 2 - 10 * cos(2*pi*x) + 10) ;
%Ackley
ackley = @(x) -20 * exp(-0.2 * sqrt( sum(x .* x)/ numel(x))) ...
        - exp(sum(cos(2*pi*x)) / numel(x)) ...
        + 20 + 2.7182818284590452353602874713526625;
%Griewank
griewank = @(x) sum(x.^2) ./ 4000 - prod(cos(x ./ sqrt([1:numel(x)]))) + 1;
%Step function 
stepfunction = @(x) sum(floor(x+0.5) .* floor(x+0.5));
%Spere
sphere = @(x) sum(x.*x);

runsconfs = containers.Map;

%Parabola
parabolaconf = struct();
parabolaconf.task.name = "Noisy Parabola";
parabolaconf.lb = [-30 -30];   parabolaconf.ub = [30 30];
parabolaconf.plb = [-20 -20];  parabolaconf.pub = [20 20];
parabolaconf.f = @quadnoisy;
parabolaconf.make.noisy = false; parabolaconf.noisy = true;
runsconfs('parabola') = parabolaconf;

%Rosenbrocks
rosenbrocksconf = struct();
rosenbrocksconf.task.name = "Rosenbrock";
rosenbrocksconf.lb = [-20 -30];   rosenbrocksconf.ub = [20 20];
rosenbrocksconf.plb = [-5 -5];  rosenbrocksconf.pub = [5 5];
rosenbrocksconf.f = @rosenbrocks;
rosenbrocksconf.make.noisy = true; rosenbrocksconf.noisy = true;
runsconfs('rosenbrocks') = rosenbrocksconf;

% Ackley 
ackleyconf = struct();
ackleyconf.task.name = 'Ackley';
ackleyconf.lb = [-32 -32];   ackleyconf.ub = [32 32];
ackleyconf.plb = [-32 -32];  ackleyconf.pub = [32 32];
ackleyconf.f = ackley;
ackleyconf.make.noisy = true; ackleyconf.noisy = true;
runsconfs('ackley') = ackleyconf;

% Rastrigin 
rastriginconf = struct();
rastriginconf.task.name = 'Rastrigin';
rastriginconf.lb = [-20 -20];   rastriginconf.ub = [20 20];
rastriginconf.plb = [-5.12 -5.12];  rastriginconf.pub = [5.12 5.12];
rastriginconf.f = rastrigin;
rastriginconf.make.noisy = true; rastriginconf.noisy = true;
runsconfs('rastrigin') = rastriginconf;

% Griewank
griewankconf = struct();
griewankconf.task.name = 'Griewank';
griewankconf.lb = [-600 -600];   griewankconf.ub = [600 600];
griewankconf.plb = [-600 -600];  griewankconf.pub = [600 600];
griewankconf.f = griewank;
griewankconf.make.noisy = true; griewankconf.noisy = true;
runsconfs('griewank') = griewankconf;

% StyblinkyTang 
styblinskytangconf = struct();
styblinskytangconf.task.name = 'Styblinskytang';
styblinskytangconf.lb = [-5 -5];   styblinskytangconf.ub = [5 5];
styblinskytangconf.plb = [-5 -5];  styblinskytangconf.pub = [5 5];
styblinskytangconf.f = @stybtang;
styblinskytangconf.make.noisy = true; styblinskytangconf.noisy = true;
runsconfs('styblinskytang') = styblinskytangconf;

% Cliff
cliffconf = struct();
cliffconf.task.name = 'Cliff';
cliffconf.lb = [-20 -20];   cliffconf.ub = [20 20];
cliffconf.plb = [-20 -20];  cliffconf.pub = [20 20];
cliffconf.f = cliff;cliffconf
cliffconf.make.noisy = true; cliffconf.noisy = true;
runsconfs('cliff') = cliffconf;

% Sphere
sphereconf = struct();
sphereconf.task.name = 'Sphere';
sphereconf.lb = [-20 -20];   sphereconf.ub = [20 20];
sphereconf.plb = [-20 -20];  sphereconf.pub = [20 20];
sphereconf.f = sphere;
sphereconf.make.noisy = true; sphereconf.noisy = true;
runsconfs('sphere') = sphereconf;

% StepFunction
stepconf = struct();
stepconf.task.name = 'Stepfunction';
stepconf.lb = [-20 -20];   stepconf.ub = [20 20];
stepconf.plb = [-20 -20];  stepconf.pub = [20 20];
stepconf.f = stepfunction;
stepconf.make.noisy = true; stepconf.noisy = true;
runsconfs('stepfunction') = stepconf;

%%
for key = keys(runsconfs)
    run.conf = runsconfs(key{1});
    if run.conf.make.noisy == true
        run.conf.f = @(x) run.conf.f(x) + randn(size(x,1),1);
        run.conf.task.name = strcat("Noisy ", run.conf.task.name);
    end
    X0 = load(strcat('./X0_examples/runs_20/x0_', run.conf.task.name,'.mat')).X0;
    seed = 3;
    res = [];
    for i = 1:size(X0, 1)
        
        x0 = X0(i, :);
        rng(seed, 'twister')
        [x,fval,exitflag,output, optimState] = bads(run.conf.f,x0,run.conf.lb,run.conf.ub, ...
                                                        run.conf.plb,run.conf.pub);
        out = struct();
        out.x = x;
        out.fval = fval;
        out.fsd = optimState.fsd;
        out.iterList.fval = optimState.iterList.fval;
        out.iterList.fsd = optimState.iterList.fsd;
        out.iterList.u = optimState.iterList.u;
        out.iterList.x = optimState.iterList.x;
        out.iterList.yval = optimState.iterList.yval;
        out.iterList.hyp = optimState.iterList.hyp;
        out.iterList.funccount = optimState.iterList.funccount;
        out.funccount = optimState.funccount;
        res = [res out];
        seed = seed + 1;
    end
    runs_res = struct();
    runs_res.conf = run.conf;
    runs_res.res = res;
    save(strcat("./output_runs/det_runs_50/", runs_res.conf.task.name,".mat"),'runs_res');
end