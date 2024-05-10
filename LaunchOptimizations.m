% a function which optimizes models for four separate phases of the
% experiment

%% prologue

clear all
close all

%% modifiable parameters

model = 'Sliding Window Thompson sampling';
reset = false; % do we reset between sessions
group_size = 24; % number of sessions per group
save_folder = [model '/Optimisations'];
addpath(model)

%% load data and optimise chosen model

load('Data/complete data')

subjects = unique(data(:,1))';

Optimise(model, data, reset, subjects, group_size, save_folder)

rmpath(model)
