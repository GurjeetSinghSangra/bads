
function [status] = multiple_runs_example(D, runs, task_to_run, noisy, input_dir, output_path)

    %Cliff
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
    
    if any(contains(lower(task_to_run), "parabola"))
        %Parabola
        parabolaconf = struct();
        parabolaconf.task.name = "Noisy Parabola";
        parabolaconf.lb = ones(1, D) .* -30;   parabolaconf.ub =  ones(1, D) .*30;
        parabolaconf.plb =  ones(1, D) .* -20; parabolaconf.pub =  ones(1, D) .* 20;
        parabolaconf.f = @quadnoisy;
        parabolaconf.make.noisy = false; parabolaconf.noisy = true;
        runsconfs('parabola') = parabolaconf;
    end
    
    %Rosenbrocks
    if any(contains(lower(task_to_run), "rosenbrock")) || any(contains(lower(task_to_run), "all"))
        rosenbrocksconf = struct();
        rosenbrocksconf.task.name = "Rosenbrock";
        rosenbrocksconf.lb = ones(1, D) .* -20;   rosenbrocksconf.ub = ones(1, D) .* 20;
        rosenbrocksconf.plb = ones(1, D) .* -5;  rosenbrocksconf.pub = ones(1, D) .* 5;
        rosenbrocksconf.f = @rosenbrocks;
        if noisy == 1
            rosenbrocksconf.make.noisy = true; rosenbrocksconf.noisy = true;
        else
            rosenbrocksconf.make.noisy = false; rosenbrocksconf.noisy = false;
        end
        runsconfs('rosenbrocks') = rosenbrocksconf;
    end
    
    % Ackley
    if any(contains(lower(task_to_run), "ackley")) || any(contains(lower(task_to_run), "all"))
        ackleyconf = struct();
        ackleyconf.task.name = 'Ackley';
        ackleyconf.lb = ones(1, D) .* -32;   ackleyconf.ub = ones(1, D) .* 32;
        ackleyconf.plb = ones(1, D) .* -32;  ackleyconf.pub = ones(1, D) .* 32;
        ackleyconf.f = ackley;
        if noisy == 1
            ackleyconf.make.noisy = true; ackleyconf.noisy = true;
        else
            ackleyconf.make.noisy = false; ackleyconf.noisy = false;
        end
        runsconfs('ackley') = ackleyconf;
    end
    
    % Rastrigin 
    if any(contains(lower(task_to_run), "rastrigin")) || any(contains(lower(task_to_run), "all"))
        rastriginconf = struct();
        rastriginconf.task.name = 'Rastrigin';
        rastriginconf.lb = ones(1, D) .* -20; rastriginconf.ub = ones(1, D) .* 20;
        rastriginconf.plb = ones(1, D) .* -5.12; rastriginconf.pub = ones(1, D) .* 5.12;
        rastriginconf.f = rastrigin;
        if noisy == 1
            rastriginconf.make.noisy = true; rastriginconf.noisy = true;
        else
            rastriginconf.make.noisy = false; rastriginconf.noisy = false;
        end
        runsconfs('rastrigin') = rastriginconf;
    end
    
    % Griewank
    if any(contains(lower(task_to_run), "griewank")) || any(contains(lower(task_to_run), "all"))
        griewankconf = struct();
        griewankconf.task.name = 'Griewank';
        griewankconf.lb = ones(1, D) .* -600; griewankconf.ub = ones(1, D) .* 600;
        griewankconf.plb = ones(1, D) .* -600; griewankconf.pub = ones(1, D) .* 600;
        griewankconf.f = griewank;
        if noisy == 1
            griewankconf.make.noisy = true; griewankconf.noisy = true;
        else
            griewankconf.make.noisy = false; griewankconf.noisy = false;
        end
        runsconfs('griewank') = griewankconf;
    end

    % StyblinkyTang 
    if any(contains(lower(task_to_run), "styblinskytang")) || any(contains(lower(task_to_run), "all"))
        styblinskytangconf = struct();
        styblinskytangconf.task.name = 'Styblinskytang';
        styblinskytangconf.lb = ones(1, D) .* -5; styblinskytangconf.ub = ones(1, D) .* 5;
        styblinskytangconf.plb = ones(1, D) .* -5;  styblinskytangconf.pub = ones(1, D) .* 5;
        styblinskytangconf.f = @stybtang;
        if noisy == 1
            styblinskytangconf.make.noisy = true; styblinskytangconf.noisy = true;
        else
            styblinskytangconf.make.noisy = false; styblinskytangconf.noisy = false;
        end
        runsconfs('styblinskytang') = styblinskytangconf;
    end
    
    % Cliff
    if any(contains(lower(task_to_run), "cliff")) || any(contains(lower(task_to_run), "all"))
        cliffconf = struct();
        cliffconf.task.name = 'Cliff';
        cliffconf.lb = ones(1, D) .* -20;   cliffconf.ub = ones(1, D) .* 20;
        cliffconf.plb = ones(1, D) .* -20;  cliffconf.pub= ones(1, D) .* 20;
        cliffconf.f = cliff;
        if noisy == 1
            cliffconf.make.noisy = true; cliffconf.noisy = true;
        else
            cliffconf.make.noisy = false; cliffconf.noisy = false;
        end
        runsconfs('cliff') = cliffconf;
    end
    
    % Sphere
    if any(contains(lower(task_to_run), "sphere")) || any(contains(lower(task_to_run), "all"))
        sphereconf = struct();
        sphereconf.task.name = 'Sphere';
        sphereconf.lb = ones(1, D) .* -20;   sphereconf.ub = ones(1, D) .* 20;
        sphereconf.plb = ones(1, D) .* -20;  sphereconf.pub = ones(1, D) .* 20;
        sphereconf.f = sphere;
        if noisy == 1
            sphereconf.make.noisy = true; sphereconf.noisy = true;
        else
            sphereconf.make.noisy = false; sphereconf.noisy = false;
        end
        runsconfs('sphere') = sphereconf;
    end
    
    % StepFunction
    if any(contains(lower(task_to_run), "stepfunction")) || any(contains(lower(task_to_run), "all"))
        stepconf = struct();
        stepconf.task.name = 'Stepfunction';
        stepconf.lb = ones(1, D) .* -20;   stepconf.ub = ones(1, D) .* 20;
        stepconf.plb = ones(1, D) .* -20;  stepconf.pub = ones(1, D) .* 20;
        stepconf.f = stepfunction;
        if noisy == 1
            stepconf.make.noisy = true; stepconf.noisy = true;
        else
            stepconf.make.noisy = false; stepconf.noisy = false;
        end
        runsconfs('stepfunction') = stepconf;
    end
    
    for key = keys(runsconfs)
        run.conf = runsconfs(key{1});
        if run.conf.make.noisy == true
            run.conf.f = @(x) run.conf.f(x) + randn(size(x,1),1);
            run.conf.task.name = strcat("Noisy ", run.conf.task.name);
        end
        X0 = load(strcat(input_dir, run.conf.task.name,'.mat')).X0; %'./X0_examples/runs_50/x0_'
        seed = 3;
        res = [];
        fprintf("Starting benchmark for ")
        fprintf(run.conf.task.name)
        fprintf("\n")
        for i = 1:runs
            
            if i <= length(X0)

                x0 = X0(i, :);
                rng(seed, 'twister')
                fprintf("Iteration %i, seed %i \n", i, seed);
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
        end
        runs_res = struct();
        runs_res.conf = run.conf;
        runs_res.res = res; % "./output_runs/det_runs_50/"
        save(strcat(output_path, runs_res.conf.task.name,".mat"),'runs_res');
    end
    status = 'finished';
end