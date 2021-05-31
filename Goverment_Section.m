%% Figures for Goverment Section: 
clear all; close all; 


%% Capital Gains Tax
% Baseline CDFs
% [1 1]
load('cdf_baseline11.mat');


% 5
load('cdf_55.mat');

% 10 
load('cdf_10.mat');

% 15
load('cdf_15.mat');


figure; 
plot(avec, cdf_pr_taur_11, 'b', 'LineWidth', 3); hold on; xlim([0 140]); ylim([0 0.5]); 
plot(avec, cdf_upr_taur_11, 'LineWidth', 3); 
plot(avec, cdf_upr_taur_5, 'LineWidth', 3); 
plot(avec, cdf_upr_taur_10, 'LineWidth', 3); 
plot(avec, cdf_upr_taur_15, 'LineWidth', 3); 
legend('Productive Baseline', 'Unproductive Baseline', 'Unproductive G = 5', '"" G = 10', ' "" G = 15')

xlabel('Assets Level'); 
title('Cumulative Distribution function for each agent and Transfers'); 


%% Wealth Tax
clear all; close all; 

load('cdfa_11.mat');
load('cdfa_5.mat'); 
load('cdfa_10.mat'); 
load('cdfa_15.mat'); 


figure; 
plot(avec, cdf_pr_taua_11, 'b', 'LineWidth', 3); hold on; xlim([0 165]); ylim([0 0.5]); 
plot(avec, cdf_upr_taua_11, 'LineWidth', 3); 
plot(avec, cdf_upr_taua_5, 'LineWidth', 3); 
plot(avec, cdf_upr_taua_10, 'LineWidth', 3); 
plot(avec, cdf_upr_taua_15, 'LineWidth', 3); 
legend('Productive Baseline', 'Unproductive Baseline', 'Unproductive G = 5', '"" G = 10', ' "" G = 15')

xlabel('Assets Level'); 
title('Cumulative Distribution function for each agent and Transfers'); 







