%% UNPRODUCTIVE AGENT CAPITAL TAX
% Full Specified Life Cycle profile
% Full specified HGeterogeneous returns  R = [1.1 , 1.03, 1.05, 1.01 ]; 
% Operative Borrowing constraint a = 0
% Now (Wednesday Afternoon) the model generates 20% of the population with 0 wealth, 
% AmaxFact = 30

clear all; close all; 
disp('-----------------------------------------------------------------------')
disp('SOLVING LIFE-CYCLE MODEL BY DYNAMIC PROGRAMMING:')

%% Parameters: 
T = 70;
retirement = 45; 
gamma = 2;  
beta = 0.98;      % Discount factor
abar = 0;         % Borrowing limit
b = 0.9;          %  Retirement Benefit
G = [1,1.1];  % Lump Sum Transfer 
taur = 0.25; 

%Life Cycle Earnings: 
phi1 = 0.04; 
phi2 = -0.0006; 
tworkvec = 1:1:retirement;
tvec = 1:1:T; 
tretvec = ones(1,25); 

yvec = [exp(phi1*(tworkvec)+phi2*(tworkvec).^2), b*tretvec]; 
% Uncertain Element: 
R = [1+0.1*(1-taur), 1+0.01*(1-taur)]; 
Ny = numel(R); 
rTrMat = [  0.3, 0.7; 0.2, 0.8]; 

%% Algorithm Parameters

Na = 1000; 
aMaxFact = 85; 

topA = aMaxFact*max(yvec); 
avec = linspace(abar, topA, Na); 
%% Value Function Iteration: 

% 3-D Arrays to storage everything
V = zeros(Na,Ny,T); 
ap = zeros(Na,Ny,T);
ai = zeros(Na,Ny,T);

c  = zeros(Na,Ny,T);
%S  = zeros(Na,Ny,T);

n  = zeros(Na,Ny,T);

aMaxInd = zeros(1,T); 

Vfin = zeros(Na,Ny);

aTrMat = cell(Ny,T);

% Create 3D array u that gives us utility when choosing any asset choice a'
% at any state (a,y) today:

aToday = reshape(avec,[1,1,Na]); 
aTomw  = avec;
apvec = reshape(avec,[1,1,Na]); 

vv = Vfin; 
ss = zeros(Na,Ny);

%% The loop: 
disp('Value-Function iteration...')
for t = T:-1:1
    
    cons = aToday + yvec(t) - aTomw'./R + G;
    NotFeas = (cons<0);
    cons(NotFeas) = 1;
    
     if t == T 
       
       NotFeas = (cons>aToday+yvec(t));
     end 
    
    u = cons.^(1-gamma)/(1-gamma) + G ;
    u(NotFeas) = -Inf;
    
    
    Wt = vv*rTrMat';
    
    Jt = u + beta*Wt;
    
    [vm,ii] = max(Jt,[],1);
    
    vv = (squeeze(vm))';
    V(:,:,t) = vv;
    ii = (squeeze(ii))';
    
    ai(:,:,t) = ii;
    aapp = avec(ii);
    ap(:,:,t) = aapp; 
    c( :,:,t) = avec' + yvec(t) - aapp./R ;   % avec is transposed with respect to the original model. 
     
    
    
    for iy=1:Ny 
        aTrMat{iy,t} = sparse((1:Na)',ii(:,iy),ones(Na,1),Na,Na);
    end 
    
    constr = (ii==1);
    ss = ss*rTrMat';
    
    for iy=1:Ny
    ss(:,iy) = constr(:,iy) + aTrMat{iy,t}*ss(:,iy);
    end 
    
    S(:,:,t) = ss;
end 



%% Density Loop: 
disp('Calculating distribution...')

nn = zeros(Na,Ny);
[a0,iii] = min(abs(avec));
nn(iii,1) = 1;
mm = iii;
qtl = [0.1,0.25,0.5,0.75,0.95];
Nq = numel(qtl);
q  = zeros(Nq,T);

for t=1:T                           % Loop over time.
    n(:,:,t) = nn;                  % Store density in beginning of period.
    for iy=1:Ny                     % Loop over all interest levels.
        nn(:,iy) = aTrMat{iy,t}'*nn(:,iy);
    end                             % Apply law of motion for assets for 
                                    % each productivity group.
    nn = nn*rTrMat;                 % Apply law of motion for income y.
    mm = max( ai(mm,:,t) );         % New highest asset index in economy:
                                    % Take old one, then take highest index
                                    % that agents reach from all prod.
                                    % levels at that age. Scalar.
    aMaxInd(t) = mm;                % Store this highest index.
    qq = GetQuantGridVar2(avec,sum(nn,2),qtl);
    q(:,t) = qq';
    
end


%%% Asset distribution for everyone 
natMarg = squeeze(sum(n,2));        % Sum over income dimension to get 
                                    % marginal distribution of assets over
                                    % time: Na-by-T matrix.
naMarg  = sum(natMarg,2);           % Sum again over time to get marginal
                                    % distribution over assets:
                                    % Na-by-1 vector.
ntMarg  = sum(natMarg,1);           % Marginal distribution over time: This
                                    % should be all ones -- check this
                                    % really is true!
                                    % 1-by-T vector.
nytMarg = squeeze(sum(n,1));        % Sum distribution over assets to get
                                    % distribution over income and age:
                                    % Ny-by-T matrix.
nyMarg  = sum(nytMarg,2);           % Sum out over time again to get 
                                    % marginal distribution over income
                                    % states: Ny-by-1 vector.

                                    
                                    
                                    
%%                                    
figure;                             % Make figure with distributions:
nRow = 3;                           % Number of supblots.

subplot(nRow,1,1)
plot(avec,cumsum(naMarg)/T);        % Get using naMarg cdf of assets.
xlabel('a: assets')
ylabel('cdf')
title('Overall asset distribution')




subplot(nRow,1,2)                   % Wealth distribution over age:
aMax = avec(aMaxInd)';              % Get maximal assets level attained at
                                    % each age: 1-by-T vector.
aTopVec = topA.*ones(1,T);          % 1-by-T vector with top asset grid 
                                    % point.
plot(tvec,[q;(aMax)';aTopVec]);         % Plot the five quantiles and the two 
                                    % vectors just created over age t.
%plot(tvec,q)                                    
lbl = cell(1,Nq+2);                 % Generate cell with labels for 
for iq=1:Nq                         % the different series:
    lbl{iq} = sprintf('Q%1.0f',100*qtl(iq));    
end                                 % Write in Q10, Q25 etc. for quantiles.
lbl{Nq+1} = 'max';                  % 2nd-to-last entry: 'max'.
lbl{Nq+2} = 'grid boundary';        % Last entry: top grid point.
legend(lbl{:})                      % Use cell to make legend.
xlabel('t: age')
ylabel('a: assets')
title('Quantiles of assets by age')

subplot(nRow,1,3)                   % Plot distribution of agents over
surf(nytMarg)
%surf(tvec,yvec,nytMarg)             % income states y age with a surface
xlabel('t: age')                    % plot.
ylabel('r: income group')
title('Distribution over income by age')








%% Compute Moments (mean and sderr Consumption)
% Example for how to calculate moments:
ctMean = sum(sum(c.*n,1),2);        % Sum over first and second dimension of
                                    % of c (weighing by the distribution)
                                    % to get mean consumption at each
                                    % age (recall that mass sums up to 1 at
                                    % each age): (1,1,T) array.
SSR = sum( sum(  n.*(c-ctMean).^2, 1), 2);  % Same for standard deviation:                               
ctStDev = sqrt( SSR );   

figure; 
plot(tvec(:),ctMean(:), 'LineWidth', 3 ); % Plot mean and standard deviation
hold on; 
plot(tvec(:), ctStDev(:), 'LineWidth', 3); 

title('Unproductive')       % of consumption. Note: 'ctMean(:)'
legend('mean','st.dev.')                   % vectorizes ctMean and gives T-by-1
xlabel('Age')                              % vector. 
ylabel('Consumption')


