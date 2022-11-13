%BADS_EXAMPLES Examples for Bayesian Adaptive Direct Search
%
%  Example 1: Basic usage
%  Example 2: Non-bound constraints
%  Example 3: Noisy objective function
%  Example 4: Extra noisy objective function
%  Example 5: Periodic function
%  Example 6: Extended usage
%
%  Note: after installation, run 
%    bads('test') 
%  to check that everything is working correctly.
%
%  For any question, check out the FAQ: 
%  https://github.com/lacerbi/bads/wiki
%
%  See also BADS.

% Luigi Acerbi 2017

display('Running a number of examples usage for Bayesian Adaptive Direct Search (BADS).');
display('Open ''bads_examples.m'' to see additional comments and instructions.');


%% Example 1: Basic usage

% Simple usage of BADS on Rosenbrock's banana function in 2D
% (see https://en.wikipedia.org/wiki/Rosenbrock_function).
% 
% We specify wide hard bounds and tighter plausible bounds that (hopefully) 
% contain the solution. Plausible bounds represent your best guess at 
% bounding the region where the solution might lie.

x0 = [0 0 0];                 % Starting point
lb = [-20 -20 -20];             % Lower bounds
ub = [20 20 20];               % Upper bounds
plb = [-5 -5 -5];              % Plausible lower bounds
pub = [5 5 5];                % Plausible upper bounds

% Screen display
fprintf('\n');
display('*** Example 1: Basic usage');
display('  Simple usage of BADS on <a href="https://en.wikipedia.org/wiki/Rosenbrock_function">Rosenbrock''s banana function</a> in 2D.');
display('  Press any key to continue.'); fprintf('\n');
pause;

% Run BADS, which returns the minimum X and its value FVAL.
[x,fval] = bads(@rosenbrocks,x0,lb,ub,plb,pub)

display('The true global minimum is at X = [1,1], where FVAL = 0.');

% Note that BADS by default does not aim for extreme numerical precision 
% (e.g., beyond the 2nd or 3rd decimal place), since in realistic 
% model-fitting problems such a resolution is typically pointless.


%% Example 2: Non-bound constraints

% We test BADS by forcing the solution to be within a circle with radius 1.
% Since we know the optimization region, we set tight hard bounds around it
% to further help the search.

x0 = [0,0];                 % Starting point
lb = [-1 -1];   ub = [1 1]; % Hard bounds only

% Note that BADS will complain because the plausible bounds are not 
% specified explicitly (it will use LB and UB instead). Generally, you want
% to specify both hard and plausible bounds.

% Non-bound constraints are violated outside the unit circle
nonbcon = @(x) sum(x.^2,2) > 1;

% Note that NONBCON requires a matrix input. Suppose we want to write the 
% above case without using SUM. We would have:
% nonbcon = @(x) (x(:,1).^2 + x(:,2).^2) > 1;   % Correct
% nonbcon = @(x) (x(1).^2 + x(2).^2) > 1;       % Wrong! not matrix input

% Screen display
fprintf('\n');
display('*** Example 2: Non-bound constraints');
display('  As before, but we force the input to stay in a circle with unit radius.');
display('  BADS will complain because the plausible bounds are not specified explicitly.');
display('  Press any key to continue.'); fprintf('\n');
pause;

% Run BADS with both bound and non-bound constraints
[x,fval] = bads(@rosenbrocks,x0,lb,ub,[],[],nonbcon)

% Alternatively, the following instructions would make BADS happier 
% (in a realistic model-fitting scenario, we recommend, whenever possible, 
% to specify plausible bounds which are tighter than the hard bounds).
%
% plb = lb; pub = ub;
% [x,fval] = bads(@rosenbrocks,x0,lb,ub,plb,pub,nonbcon)

display('The true global minimum under these constraints is at X = [0.786,0.618], where FVAL = 0.046.');


%% Example 3: Noisy objective function

% We test BADS on a noisy function. For the purpose of this test, we
% manually add unit Gaussian noise to the 'sphere' (quadratic) function.

% Define the noisy objective as a function handle
noisyfun = @(x) sum(x.^2,2) + randn(size(x,1),1);
x0 = [-3 -3];       % For a change, we start farther away from the solution
% This time to help the search we set tighter bounds
lb = [-5 -5];   ub = [5 5];
plb = [-2 -2];  pub = [2 2];

% Screen display
fprintf('\n');
display('*** Example 3: Noisy objective function');
display('  We test BADS on a noisy quadratic function with unit Gaussian noise.');
display('  Press any key to continue.'); fprintf('\n');
pause;

% Run BADS on the noisy function
[x,fval,exitflag,output] = bads(noisyfun,x0,lb,ub,plb,pub);
x
fval
display(['The true, noiseless value of the function at X is ' num2str(sum(x.^2,2)) '.']);
display('The true global minimum is at X = [0,0], where FVAL = 0.');

% FVAL in this case is an *estimate* of the function at X, obtained by
% averaging ten function evaluations. These values can be found in the
% OUTPUT structure, together with additional information about the optimization.

display('The returned OUTPUT structure is:');
output

% Note that the fractional overhead of BADS reported in OUTPUT is astronomical.
% The reason is that the objective function we are using is analytical and 
% extremely fast, which is not what BADS is designed for. 
% In a realistic scenario, the objective function will be moderately costly
% (e.g., more than 0.1 s per function evaluation), and the fractional 
% overhead should be less than 1.


%% Example 4: Extra noisy objective function

% We test BADS on a particularly noisy function and look at some options.

% Define noisy objective with substantial input-dependent noise
noisyfun = @(x) sum(x.^2,2) + (3 + 0.1*sqrt(sum(x.^2,2))).*randn(size(x,1),1);

% For this optimization, we explicitly tell BADS that the objective is
% noisy (it is not necessary, but it is a good habit); and also specify a 
% rough estimate for the value of the noise in a neighborhood of the solution.
% Finally, we tell BADS to use more samples to estimate FVAL at the end.

options = [];                       % Reset the OPTIONS struct
options.UncertaintyHandling = 1;    % Tell BADS that the objective is noisy
options.NoiseSize           = 5;    % Estimate of noise
options.NoiseFinalSamples   = 100;  % # samples to estimate FVAL at the end 
                                    % (default would be 10)
x0 = [-3 -3];
lb = [-5 -5];   ub = [5 5];
plb = [-2 -2];  pub = [2 2];

% Screen display
fprintf('\n');
display('*** Example 4: Extra noisy function');
display('  We test BADS on a particularly noisy function.');
display('  Press any key to continue.'); fprintf('\n');
pause;

% Run BADS on the noisy function
[x,fval,exitflag,output] = bads(noisyfun,x0,lb,ub,plb,pub,[],options);
x
fval
display(['The true, noiseless value of the function at X is ' num2str(sum(x.^2,2)) '.']);
display('The true global minimum is at X = [0,0], where FVAL = 0.');
display('Due to the elevated level of noise, we do not necessarily expect high precision in the solution.');


%% Example 5: Objective function with periodic dimensions

% We test BADS on a function with a subset of periodic dimensions.

% This function is periodic along the third and fourth dimension, with
% periods respectively 4 and 2.
periodicfun = @(x) rosenbrocks(x(:,1:2)) + cos(x(:,3)*pi/2) + cos(x(:,4)*pi) + 2;

x0 = [-3 -3 -1 -1];
% We specify the periodic bounds via hard bounds
lb = [-10 -5 -2 -1];
ub = [5 10 2 1];

plb = [-2 -2 -2 -1];
pub = [2 2 2 1];

options = [];                       % Reset the OPTIONS struct
options.PeriodicVars = [3 4];       % The 3rd and 4th variables are periodic

% Screen display
fprintf('\n');
display('*** Example 5: Objective function with periodic dimensions');
display('  We test BADS on a function with some periodic inputs.');
display('  Press any key to continue.'); fprintf('\n');
pause;

[x,fval,exitflag,output] = bads(periodicfun,x0,lb,ub,plb,pub,[],options);
x
fval
display('The true global minimum is at X = [1,1,±2,±1], where FVAL = 0.');


%% Example 6: Extended usage

% Extended usage of BADS that shows some additional options.

% Function handle for function with multiple input arguments (e.g., here
% we add a translation of the input; but more in general you could be 
% passing additional data to your objective function).
fun = @(x,mu) rosenbrocks(bsxfun(@plus, x, mu)); 

% This will translate the Rosenbrock fcn such that the global minimum is at zero
mu = [1 1 1 1];

% We now set bounds using also fixed variables
% (2nd and 4th variable are fixed by setting all bounds and X0 equal)

plb = [-2 0 -2 0];             % Plausible lower bounds
pub = [2 0 2 0];               % Plausible upper bounds
lb = [-20 0 -5 0];             % Hard lower bounds
ub = [20 0 5 0];               % Hard upper bounds

% Random starting point inside plausible region. In a typical optimization
% scenario, you will repeat the optimization from different starting
% points (ideally 10 or more), possibly drawn this way, and take the best
% result.
x0 = plb + (pub-plb).*rand(1,numel(plb));

options = bads('defaults');             % Get a default OPTIONS struct
options.MaxFunEvals         = 50;       % Very low budget of function evaluations
options.Display             = 'final';   % Print only basic output ('off' turns off)
options.UncertaintyHandling = 0;        % We tell BADS that the objective is deterministic

% Custom output function (return FALSE to continue, TRUE to stop optimization)
options.OutputFcn           = @(x,optimState,state) ~isfinite(fprintf('%s %d... ', state, optimState.iter));

% Screen display
fprintf('\n');
display('*** Example 6: Extended usage');
display('  Extended usage of BADS with additional options and no detailed display.');
display('  Press any key to continue.'); fprintf('\n');
pause;

% Run BADS, passing MU as additional (fixed) input argument for FUN
[x,fval,exitflag,output] = bads(fun,x0,lb,ub,plb,pub,[],options,mu);

% The following line of code would do the same using an anonymous function
% [x,fval,exitflag] = bads(@(x) fun(x,mu),x0,lb,ub,plb,pub,[],options);

x
fval
display('The true global minimum is at X = [0,0,0,0], where FVAL = 0.');
exitflag
display('EXITFLAG of 0 means that the maximum number of function evaluations has been reached.');
fprintf('\n');
display('For this optimization we used the following OPTIONS:')
options
display('Type ''help bads'' for additional documentation on BADS, or consult the <a href="https://github.com/lacerbi/bads">Github page</a> or <a href="https://github.com/lacerbi/bads/wiki">online FAQ</a>.');


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