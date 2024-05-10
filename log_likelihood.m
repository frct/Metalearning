function negLL = log_likelihood(vectParam, model, reset, data)

%log_likelihoodLU2 returns the negative log-likelihood of the model with
%parameter values given by vectParam.

%% fixed parameters

constrained = true;

%% model parameters
alpha = vectParam(1);
%beta = vectParam(2);

switch (model)
    
    case 1    
        [LL, ~] = SimulateQL(data, alpha, beta, constrained, reset);
        
    case 2
        alpha2 = alpha;
        [LL, ~] = SimulateForgettingQL(data, alpha, beta, alpha2, constrained, reset);
        
    case 3
        alpha2 = vectParam(3);
        [LL, ~] = SimulateForgettingQL(data, alpha, beta, alpha2, constrained, reset);
        
    case 4
        alpha_cf = alpha;
        counterfactual = 'win';
        [LL, ~] = CounterFactualQL(data, alpha, beta, alpha_cf, constrained, reset, counterfactual);
        
    case 5
        alpha_cf = vectParam(3);
        counterfactual = 'win';
        [LL, ~] = CounterFactualQL(data, alpha, beta, alpha_cf, constrained, reset, counterfactual);
        
    case 6
        alpha_cf = alpha;
        counterfactual = 'lose';
        [LL, ~] = CounterFactualQL(data, alpha, beta, alpha_cf, constrained, reset, counterfactual);
        
    case 7
        alpha_cf = vectParam(3);
        counterfactual = 'lose';
        [LL, ~] = CounterFactualQL(data, alpha, beta, alpha_cf, constrained, reset, counterfactual);
        
    case 8
        alpha_cf = alpha;
        counterfactual = 'all';
        [LL, ~] = CounterFactualQL(data, alpha, beta, alpha_cf, constrained, reset, counterfactual);
        
    case 9
        alpha_cf = vectParam(3);
        counterfactual = 'all';
        [LL, ~] = CounterFactualQL(data, alpha, beta, alpha_cf, constrained, reset, counterfactual);
        
    case 10
        alpha2 = min(1, vectParam(3) * alpha);
        [LL, ~] = SimulateForgettingQL(data, alpha, beta, alpha2, constrained, reset);
        
    case 11
        beta0 = beta;
        alpha2 = vectParam(3);
        k = vectParam(4);
        a = vectParam(5);
        [LL, ~] = LogIncrease(data, alpha, beta0, alpha2, k, a, constrained, reset);
    
    case 12
        beta0 = beta;
        alpha2 = vectParam(3);
        beta_m = vectParam(4);
        a = vectParam(5);
        [LL, ~] = GeomIncrease(data, alpha, beta0, alpha2, beta_m, a, constrained, reset); 
        
    case 13
        beta1 = beta;
        alpha2 = vectParam(3);
        beta2 = vectParam(4);
        beta3 = vectParam(5);
        beta4 = vectParam(6);
        [LL, ~] = StaggeredBeta(data, alpha, beta1, beta2, beta3, beta4, alpha2, constrained, reset);
        
    case 14
        beta0 = beta;
        alpha2 = vectParam(3);
        beta1 = vectParam(4);
        alpha_R = vectParam(5);
        [LL, ~] = BetaML(data, alpha, beta0, alpha2, beta1, alpha_R, constrained, reset);
        
    case 15
        betaM = beta;
        alpha2 = vectParam(3);
        k = vectParam(4);
        alpha_R = vectParam(5);
        [LL, ~] = SigmoidML(data, alpha, betaM, alpha2, k, alpha_R, constrained, reset);
    
    case 16
        beta_min = beta;
        alpha2 = vectParam(3);
        beta_max = vectParam(4);
        k = vectParam(5);
        alpha_R = vectParam(6);
        [LL, ~] = SigmoidMLbis(data, alpha, beta_min, alpha2, beta_max, k, alpha_R, constrained, reset);
        
    case 17
        beta_max = beta;
        alpha2 = vectParam(3);
        k = vectParam(4);
        alpha_R = vectParam(5);
        R50 = vectParam(6);
        [LL, ~] = Sigmoid_with_R50(data, alpha, beta_max, alpha2, k, alpha_R, R50, constrained, reset);
       
    case 'linear alpha ML'
        alpha_min = alpha;
        alpha2 = vectParam(3);
        alpha_max = vectParam(4);
        alpha_R = vectParam(5);
        [LL, ~] = alpha_linear(data, alpha_min, beta, alpha2, alpha_max, alpha_R, constrained, reset);
        
    case 'linear alpha2 ML'
        alpha2_min = vectParam(3);
        alpha2_max = vectParam(4);
        alpha_R = vectParam(5);
        [LL, ~] = alpha2_linear(data, alpha, beta, alpha2_min, alpha2_max, alpha_R, constrained, reset);
        
    case 'linear time'
        beta_min = beta;
        alpha2 = vectParam(3);
        beta_max = vectParam(4);
        [LL, ~] = beta_linear_time(data, alpha, beta_min, alpha2, beta_max, constrained, reset);
        
    case 'staggered alpha'
        alpha1 = alpha;
        alpha_forget = vectParam(3);
        alpha2 = vectParam(4);
        alpha3 = vectParam(5);
        alpha4 = vectParam(6);
        [LL, ~] = StaggeredAlpha(data, alpha1, beta, alpha2, alpha3, alpha4, alpha_forget, constrained, reset);
        
    case 'staggered alpha2'
        alpha_f1 = vectParam(3);
        alpha_f2 = vectParam(4);
        alpha_f3 = vectParam(5);
        alpha_f4 = vectParam(6);
        [LL, ~] = StaggeredAlpha2(data, alpha, beta, alpha_f1, alpha_f2, alpha_f3, alpha_f4, constrained, reset);
        
    case 'linear time alpha'
        alpha_min = alpha;
        alpha2 = vectParam(3);
        alpha_max = vectParam(4);
        [LL, ~] = alpha_linear_time(data, alpha_min, beta, alpha2, alpha_max, constrained, reset);
        
    case 'Dynamic Thompson sampling'
        C = vectParam(1);
        [LL,~] = DynamicThompsonSampling(data, C, constrained);
        
    case 'Sliding Window Thompson sampling'
        T = vectParam(1);
        [LL,~] = SlidingWindowThompsonSampling(data, T, constrained);
end
    
negLL = - LL;
end