function [loglikelihood, Qvalues, proba, RPE, trial_logL, simulated_actions, simulated_rwds] = SimulateForgettingQL(data, alpha, beta, alpha2, constrained, reset)

n_trials = size(data,1);
n_actions = 3;

Qvalues = zeros(n_trials+1,n_actions);
proba = zeros(n_trials,n_actions);
RPE = zeros(n_trials,n_actions);
simulated_actions = zeros(n_trials,1); % choices made by the model
simulated_rwds = zeros(n_trials,1); % rewards obtained by the model
trial_logL = zeros(n_trials,1);

probaRwd = [ 7/8 1/16 ; 5/8 3/16 ]; % BR 1 0 ; HR 1 0


for t=1:n_trials
    % the choice of the model
    proba(t,:) = exp(beta * Qvalues(t,:)) / sum(exp(beta * Qvalues(t,:)));
    
    %bestaction = data(t,5);
    
    if constrained % use the experimental data to update QL
        action = data(t,5);
        reward = data(t,7);
        trial_logL(t) = log(proba(t,action));
    else 
        best_action = data(t, 4);
        risk = data(t,6)+1;
        action = randsample(1:n_actions, 1, true, proba(t,:));
        reward = (action==best_action) * (drand01([(1-probaRwd(risk,1)) probaRwd(risk,1)])-1) + (action~=best_action) * (drand01([(1-probaRwd(risk,2)) probaRwd(risk,2)])-1);
        simulated_actions(t) = action;
        simulated_rwds(t) = reward;
    end
    
    RPE(t,action) = reward - Qvalues(t, action);
    
    if t < n_trials
        if reset && data(t+1,2) ~= data(t,2) % reset Q values
            Qvalues(t+1, :) = 0;
        else % if the next trial belongs to the same session, apply forgetting QL rule
            Qvalues(t+1,:) = (1-alpha2) * Qvalues(t,:);
            Qvalues(t+1, action) = Qvalues(t, action) + alpha * RPE(t,action);            
        end
    end
end

loglikelihood = sum(trial_logL);