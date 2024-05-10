function Optimise(model, data, reset, subjects, group_size, save_folder)
%my_fmcLL performs an optimization of parameters of model minimizing the
%negative log likelihood with a gradient descent method (fmincon)
%initialized at a series of starting points

n_groups = 24 / group_size;

switch (model)
    
    case 1 % QL
        mygrid = [0.1 0.5 0.9; 1 5 20];  % alpha, beta
        maxParam = [1 Inf];
        minParam = [0 0];
        
    case 2 % QL with forgetting rate equal to learning rate
        mygrid = [0.1 0.5 0.9; 1 5 20];  % alpha, beta
        maxParam = [1 Inf];
        minParam = [0 0];
        
    case 3% QL with distinct forgetting rate
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9];  % alpha, beta, alpha2
        maxParam = [1 Inf 1];
        minParam = [0 0 0];
        
    case 4 % counterfactual QL on win trials
        mygrid = [0.1 0.5 0.9; 1 5 20]; %alpha, beta
        maxParam = [1 Inf];
        minParam = [0 0];
        
    case 5 % counterfactual QL on win trials with distinct learning rate
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9]; %alpha, beta, alpha_cf
        maxParam = [1 Inf 1];
        minParam = [0 0 0];
        
    case 6 % counterfactual QL on lose trials
        mygrid = [0.1 0.5 0.9; 1 5 20]; %alpha, beta
        maxParam = [1 Inf];
        minParam = [0 0];
        
    case 7 % counterfactual QL on lose trials with distinct learning rate
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9]; %alpha, beta, alpha_cf
        maxParam = [1 Inf 1];
        minParam = [0 0 0];
        
    case 8 % counterfactual QL on all trials
        mygrid = [0.1 0.5 0.9; 1 5 20]; %alpha, beta
        maxParam = [1 Inf];
        minParam = [0 0];
        
    case 9 % counterfactual QL on all trials with distinct learning rate
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9]; %alpha, beta, alpha_cf
        maxParam = [1 Inf 1];
        minParam = [0 0 0];
        
    case 10 % forgetting QL with forgetting rate proportional to learning rate
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.8 1 1.2];  % alpha, beta, Gain
        maxParam = [1 Inf Inf];
        minParam = [0 0 0];
        
    case 11
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 0.1 0.5 2; 0.01 0.05 0.1];  % alpha, beta0, alpha2, k, a
        maxParam = [1 Inf 1 Inf Inf];
        minParam = [0 0 0 0 0];
        
    case 12
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 1 5 20; 0.01 0.05 0.1];  % alpha, beta0, alpha2, beta_m, a
        maxParam = [1 Inf 1 Inf 1];
        minParam = [0 0 0 0 0];
        
    case 13
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 1 5 20; 1 5 20; 1 5 20];
        maxParam = [1 Inf 1 Inf Inf Inf];
        minParam = [0 0 0 0 0 0];
        
    case 14
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9]; % alpha, beta0, alpha2, beta1, alpha_R
        maxParam = [1 Inf 1 Inf 1];
        minParam = [0 0 0 0 0];
        
    case 15
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9]; % alpha, beta_max, alpha2, k, alpha_R
        maxParam = [1 Inf 1 Inf 1];
        minParam = [0 0 0 0 0];
        
    case 16
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 1 5 20; 1 5 20; 0.1 0.5 0.9]; % alpha, beta_min, alpha2, beta_max, k, alpha_R
        maxParam = [1 Inf 1 Inf Inf 1];
        minParam = [0 0 0 0 0 0];
        
    case 17 % sigmoid model with free R50 and beta_min = 0
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 0.1 0.5 0.9]; % alpha, beta_max, alpha2, k, alpha_R, R50
        maxParam = [1 Inf 1 Inf 1 Inf];
        minParam = [0 0 0 0 0 0];
        
    case 'linear alpha ML'
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 0.1 0.5 0.9; 0.1 0.5 0.9]; % alpha_min, beta, alpha2, alpha_max, alpha_R
        maxParam = [1 Inf 1 1 1];
        minParam = [0 0 0 0 0];
        
    case 'linear alpha2 ML'
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 0.1 0.5 0.9; 0.1 0.5 0.9]; % alpha, beta, alpha2_min, alpha2_max, alpha_R
        maxParam = [1 Inf 1 1 1];
        minParam = [0 0 0 0 0];
        
    case 'linear time'
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 1 5 20];  % alpha, beta0, alpha2, beta_max, a
        maxParam = [1 Inf 1 Inf];
        minParam = [0 0 0 0];
        
    case 'staggered alpha'
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 0.1 0.5 0.9; 0.1 0.5 0.9; 0.1 0.5 0.9;];
        maxParam = [1 Inf 1 1 1 1];
        minParam = [0 0 0 0 0 0];
        
    case 'staggered alpha2'
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 0.1 0.5 0.9; 0.1 0.5 0.9; 0.1 0.5 0.9;];
        maxParam = [1 Inf 1 1 1 1];
        minParam = [0 0 0 0 0 0];
        
    case 'linear time alpha'
        mygrid = [0.1 0.5 0.9; 1 5 20; 0.1 0.5 0.9; 0.1 0.5 0.9]; %alpha_min, beta, alpha2, alpha_max
        maxParam = [1 Inf 1 1];
        minParam = [0 0 0 0];
        
    case 'Dynamic Thompson sampling'
        mygrid = [1000 2000 5000];
        maxParam = [inf];
        minParam = [100];
        
    case 'Sliding Window Thompson sampling'
        mygrid = [5 10 20];
        maxParam = [100];
        minParam = [1];
end

nbparam = length(mygrid(:,1));
options = optimset('Algorithm','interior-point');
fmcResults = zeros(3^nbparam,nbparam+2);
gradient = zeros(3^nbparam,nbparam);
hessian = zeros(3^nbparam,nbparam,nbparam);

for group = 1 : n_groups
    session_data = data(data(:,2) > (group - 1) * group_size & data(:,2) <= group * group_size, :);
    for nsub = subjects
        ratid = ['Rat number ' num2str(nsub)]
        rat_data = session_data(session_data(:,1) == nsub, :);
        nt = size(rat_data, 1);
        
        M = all_cb(nbparam); % all the combinations of indices of mygrid that need to be tested
        vectParam = zeros(1, nbparam);
        
        for niter = 1 : length(M(:,1))
            iteration = [ 'iteration ' num2str(niter) ' out of ' num2str(3^nbparam)]
            for i = 1 : nbparam
                vectParam(i) = mygrid(i, M(niter,i)); % get current combination of intialization values
            end
            
            if strcmp(model, 'Dynamic Thompson sampling') % single parameter optimisation
                fun = @(x) log_likelihood(x, model, reset, rat_data);
                [x, fval, ~, ~, ~, grad, hess] = fmincon(fun, vectParam, [], [], [], [], minParam, maxParam, [], options);
                fmcResults(niter,:) = [x fval exp(-fval/nt)];
                gradient(niter,:) = grad';
                hessian(niter,:,:) = hess;
            else            
                [x, fval, ~, ~, ~, grad, hess] = fmincon(@(x) log_likelihood(x, model, reset, rat_data), vectParam, [], [], [], [], minParam(1:nbparam), maxParam(1:nbparam), [], options);
                fmcResults(niter,:) = [x fval exp(-fval/nt)];
                gradient(niter,:) = grad';
                hessian(niter,:,:) = hess;
            end
            
        end
        
        if n_groups > 1
            save([save_folder '/Rat' num2str(nsub) 'sessiongroup' num2str(group) '_fmcResults'], 'fmcResults','gradient','hessian')
        else
            save([save_folder '/Rat' num2str(nsub) '_fmcResults'], 'fmcResults','gradient','hessian')
        end
    end
    
end
