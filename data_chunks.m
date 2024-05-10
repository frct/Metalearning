% separate the data,after cleaning, into nice chunks of HR and LR blocks of
% sessions 1-6, 7-12, 13-18, 19-24

clear all
close all

load('S1-12.txt')
load('S13-24.txt')
S13_24(:,2) = S13_24(:,2) + 12; % renumber sessions properly

raw_data = [S1_12; S13_24];
% data presently has 12 columns: Rat, session, block, signal (=0), target,
% choice, risk, reward, t1, t2, t3 and condition(?)

data = raw_data(:, [1, 2, 3, 5, 6, 7, 8]); %getting rid of columns I don't need eg signal, start time, latency satiété and dose 
l = lines_to_be_deleted(data); % data now has 7 columns: Rat, session, block, target, choice, risk and reward
data(l,:) = [];

%% save

save('complete data', 'raw_data', 'data')

%% risk distinction
HR = data(data(:,6) == 1,:);
LR = data(data(:,6) == 0,:);

HR1 = HR(HR(:,2)<7,:); % first 6 sessions which are HR
HR2 = HR(HR(:,2)>6 & HR(:,2)<13,:);
HR3 = HR(HR(:,2)>12 & HR(:,2)<19,:);
HR4 = HR(HR(:,2)>18,:);

LR1 = LR(LR(:,2)<7,:);
LR2 = LR(LR(:,2)>6 & LR(:,2)<13,:);
LR3 = LR(LR(:,2)>12 & LR(:,2)<19,:);
LR4 = LR(LR(:,2)>18,:);

save('separate risk data','HR1', 'HR2', 'HR3', 'HR4', 'LR1', 'LR2', 'LR3', 'LR4')