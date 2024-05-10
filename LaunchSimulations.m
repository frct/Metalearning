% Launch simulations of ML model

clear all
close all

subjects = 27 : 50;
nbAction = 3;
simulation = true;
reset = false;
n_reps = 100;


%% Model specs and optimized parameters

model = 'linear alpha2 ML';
folder = 'ML alpha2 linear/Simulations';
load('ML alpha2 linear/Optimisations/Best parameters.mat');

%% loading behavioral data

load('Data/complete data');
data(data(:,3)>12,:) = [];

%% LAUNCH SIMULATIONS WITH OPTIMIZED PARAMS


for rat = 1:length(subjects)
    rat_id = subjects(rat);
    save_file = [folder '/Rat' num2str(rat_id) '_' num2str(n_reps) 'simulations'];
    rat_data = data(data(:,1) == rat_id, :);
    n_trials = length(rat_data(:,1));
    
    %% get parameters for this rat
    switch model
        case 'linear alpha ML'
            alpha_min = best_params(rat,1);
            beta = best_params(rat,2);
            alpha2 = best_params(rat,3);
            alpha_max = best_params(rat,4);
            alpha_R = best_params(rat,5);
            alphas = zeros(n_trials, n_reps);
            
        case 'linear alpha2 ML'
            alpha = best_params(rat,1);
            beta = best_params(rat,2);
            alpha2_min = best_params(rat,3);
            alpha2_max = best_params(rat,4);
            alpha_R = best_params(rat,5);
            alphas = zeros(n_trials, n_reps);
    end
    %% launch simulations
    Qvalues = zeros(n_trials+1, nbAction, n_reps);
    choices = zeros(n_trials, n_reps);
    rewards = zeros(n_trials, n_reps);
    probas = zeros(n_trials, nbAction, n_reps);
    constrained = false;
    reset = false;
    
    for i = 1 : n_reps
        %[loglikelihood, Q, proba, RPE, trial_logL, simulated_actions, simulated_rwds, alpha_list] = alpha_linear(rat_data, alpha_min, beta, alpha2, alpha_max, alpha_R, constrained, reset);
        [loglikelihood, Q, proba, RPE, trial_logL, simulated_actions, simulated_rwds, alpha_list] = alpha2_linear(rat_data, alpha, beta, alpha2_min, alpha2_max, alpha_R, constrained, reset);
        Qvalues(:,:,i) = Q;
        probas(:,:,i) = proba;
        choices(:,i) = simulated_actions;
        rewards(:,i) = simulated_rwds;
    end
    
    save(save_file, 'rat_data', 'Qvalues', 'probas', 'choices', 'rewards');
end