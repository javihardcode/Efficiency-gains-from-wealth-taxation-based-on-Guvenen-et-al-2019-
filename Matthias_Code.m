
%% Life-cycle model with income shocks: Dynamic programming solution.
% Algorithm with brute-force discretization in assets.
clear                               % Clear workspace.
close all                           % Close open figure windows.
% Give messages:
disp('-----------------------------------------------------------------------')
disp('SOLVING LIFE-CYCLE MODEL BY DYNAMIC PROGRAMMING:')

%% Model parameters:
T  =  45;                           % Number of time periods.
gamma = 2;                          % Curvature of power utility.
beta  = 0.98;                       % Discount factor.
R     = 1.02;                       % Gross real interest rate.
abar  = 0;                          % Borrowing limit.

yvec  = [1, 2, 3];                  % Vector with incomes: 1-by-Ny.
Ny = numel(yvec);                   % Number of grid points for income.
yTrMat= [0.9, 0.1, 0  ; ...         % Matrix with income transition proba-
         0.1, 0.8, 0.1; ...         % bilities. Rows sum up to 1, i.e. row
         0  , 0.1, 0.9     ];       % tells us which state we come from and 
                                    % column which state we go to:
                                    % Ny-by-Ny matrix.
% Print out parameters:     
fprintf('Parameters: T=%1.0f, gamma=%1.2f, beta=%1.2f, R=%1.2f, abar=%1.2f, %1.0f income states.\n',...
        [T,gamma,beta,R,abar,Ny] );
    
%% Algorithm parameters:
Na = 1000;                          % Number of grid points for assets.
aMaxFact = 6;                       % How many times the highest income we 
                                    % make the asset grid.

topA = aMaxFact*yvec(end);          % Top asset grid point.

% Linearly-spaced grid:
avec = linspace(abar,topA,Na)';     % Vector with asset grid: Na-by-1.
% Tried log-type grid: Does not look better.
% avec = GetLogGrid(abar,topA,-yvec(1),Na)';
tvec = 1:1:T;                       % Vector with time periods: 1-by-T.
% Give message:
fprintf('Grid: %1.0f asset points (%1.2f to %1.2f).\n', ...
               [Na, avec(1), avec(Na)]                        );
           
%% Set up everything:

tic                                 % Start taking time.
disp('Value-function iteration...') % Give message what's done.
% 3D arrays (Na,Ny,T) for:
V  = zeros(Na,Ny,T);                % value function,
ap = zeros(Na,Ny,T);                % policy function: optimal savings a',
ai = zeros(Na,Ny,T);                % policy: index of the asset level that
                                    % is chosen.
c  = zeros(Na,Ny,T);                % consumption function.
S  = zeros(Na,Ny,T);                % Expected times a person is constrained:
                                    % Compute this is an example for other
                                    % quantities we may want to compute
                                    % (not being value or policy
                                    % functions).
n  = zeros(Na,Ny,T);                % Distribution function (measured when
                                    % entering period).
aMaxInd = zeros(1,T);               % Index of maximal asset level in 
                                    % economy at each age: 1-by-T.

Vfin = zeros(Na,Ny);                % Terminal value function: make it zero.

aTrMat = cell(Ny,T);                % Cell with transition matrices on
                                    % asset grid implied by optimal policies. 
                                    % Matrices will be Na-by-Na, have Ny
                                    % for each time period. Cell is
                                    % Ny-by-T.

% Create 3D array u that gives us utility when choosing any asset choice a'
% at any state (a,y) today:
aToday = reshape(avec,[1,1,Na]);    % Array with asset positions tomorrow:
                                    % (1,1,Na) array.
aTomw  = avec;                      % Today's assets: (Na,1,1) array.                                    
apvec = reshape(avec,[1,1,Na]);     % Array with asset positions tomorrow:
                                    % (1,1,Na) array.
cons = aToday + yvec - aTomw/R;     % Consumption implied by choosing each 
                                    % point in apvec given state today (a
                                    % and y): (Na,Ny,Na) array.
NotFeas = (cons<0);                 % Logical array that tells us which 
                                    % choices are not feasible.
cons(NotFeas) = 1;                  % Set consumption to positive number to
                                    % avoid complex numbers.
u = cons.^(1-gamma)/(1-gamma);      % Get utility for each choice, setting
u(NotFeas) = -Inf;                  % -Inf for infeasible choices:
                                    % (Na,Ny,Na) array.
                                    
vv = Vfin;                          % Set current value function:
                                    % (Na,Ny) matrix.
ss = zeros(Na,Ny);                  % Initialize expected time constrained:
                                    % All zeros. (Na,Ny) matrix.

                                    
%%                                    
for t=T:-1:1                        % Loop back in time.  
    Wt = vv*yTrMat';                % Apply transition matrix for income to
                                    % tomorrow's value to obtain
                                    % continuation value W_t(a,y):
                                    % (Na,Ny) matrix.
    Jt = u + beta*Wt;               % Payoff from each choice: J_t(a';a,y)
                                    % in lecture notes. Note that 
                                    % all choices leading to the same a'
                                    % have the same continuation value;
                                    % Matlab extends array cv in third
                                    % dimension just copying itself:
                                    % (Na,Ny,Na) array.
                                    % Dim.1: assets tomorrow (a').
                                    % Dim.2: income (y).
                                    % Dim.3: assets today (a).
    [vm,ii] = max(Jt,[],1);         % Pick maximum along dimension 1 (a') 
                                    % to get new value function vm:
                                    % (1,Ny,Na) array.
                                    % As second output, get indeces ii at
                                    % which maximum occurred: 
                                    % (1,Ny,Na) array.
    vv = (squeeze(vm))';            % Squeeze out first dimension and make
                                    % vm Ny-by-Na, then transpose to get
                                    % new value function vv: Na-by-Ny.
    V(:,:,t) = vv;                  % Store value function today.
    ii = (squeeze(ii))';            % Also transform indeces ii to Na-by-Ny 
                                    % matrix.
    ai(:,:,t) = ii;                 % Store index of asset grid points a' 
                                    % that are chosen at t.
    aapp = avec(ii);                % Obtain new asset positions a' by indexing
                                    % indexing vector avec with matrix ii: 
                                    % gives Na-by-Ny matrix with the new
                                    % assets in each position, which is our
                                    % policy function: Na-by-Ny matrix.
    ap(:,:,t) = aapp;               % Store this policy at t.
    c( :,:,t) = avec + yvec - aapp/R;
                                    % Obtain optimal consumption policy and
                                    % store.
   
    % Store matrices with law of motion for assets (will need them for 
    for iy=1:Ny                     % Loop over al productivities.
        aTrMat{iy,t} = sparse((1:Na)',ii(:,iy),ones(Na,1),Na,Na);
    end                             % For each productivity, create Na-by-Na
                                    % transition matrix in assets. [1,2,...,Na]
                                    % are the indeces we come from,
                                    % ii(:,iy) are the indeces we go to.
    % An example of how we can use the transition matrices to compute 
    % other variables that obey Bellman-type equations:
    constr = (ii==1);               % Create logical variable: 1 if agent 
                                    % constrained (chooses a'=abar) and 0
                                    % if unconstrained: Na-by-Ny matrix.
    ss = ss*yTrMat';                % First apply productivity shocks.
    for iy=1:Ny                     % Then, use transition matrices for a
        ss(:,iy) = constr(:,iy) + aTrMat{iy,t}*ss(:,iy); % and add 1 if 
    end                             % constrained at this t.
    S(:,:,t) = ss;                  % Store result.
end
                                    
toc                                 % Measure time taken to find value function.

%% Density iteration loop
disp('Calculating distribution...')
tic                                 % Take time.
nn = zeros(Na,Ny);                  % Create running variable for distribution.
nn(1,1) = 1;                        % Put all agents in low-a, low-y state.
                                    % Na-by-Ny matrix.
mm = 1;                             % Current maximal index: Agents start 
                                    % at first asset grid point.
qtl = [0.1,0.25,0.5,0.75,0.9];      % Which quantiles of the wealth distribution
Nq = numel(qtl);                    % we want to look at: 1-by-Nq.
q   = zeros(Nq,T);                  % Matrix with quantiles of wealth 
                                    % distribution by age.
%%
for t=1:T                           % Loop over time.
    n(:,:,t) = nn;                  % Store density in beginning of period.
    for iy=1:Ny                     % Loop over all productivity levels.
        nn(:,iy) = aTrMat{iy,t}'*nn(:,iy);
    end                             % Apply law of motion for assets for 
                                    % each productivity group.
    nn = nn*yTrMat;                 % Apply law of motion for income y.
    mm = max( ai(mm,:,t) );         % New highest asset index in economy:
                                    % Take old one, then take highest index
                                    % that agents reach from all prod.
                                    % levels at that age. Scalar.
    aMaxInd(t) = mm;                % Store this highest index.
    qq = GetQuantGridVar2(avec,sum(nn,2),qtl);
    q(:,t) = qq';
end
toc                                 % Show elapsed time for distribution.

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

figure;                             % Make figure with distributions:
nRow = 3;                           % Number of supblots.

subplot(nRow,1,1)
plot(avec,cumsum(naMarg)/T);        % Get cdf of assets.
xlabel('a: assets')
ylabel('cdf')
title('Overall asset distribution')


subplot(nRow,1,2)                   % Wealth distribution over age:
aMax = avec(aMaxInd)';              % Get maximal assets level attained at
                                    % each age: 1-by-T vector.
aTopVec = topA.*ones(1,T);          % 1-by-T vector with top asset grid 
                                    % point.
plot(tvec,[q;aMax;aTopVec])         % Plot the five quantiles and the two 
                                    % vectors just created over age t.
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
surf(tvec,yvec,nytMarg)             % income states y age with a surface
xlabel('t: age')                    % plot.
ylabel('y: income group')
title('Distribution over income by age')


%% Plot consumption functions:
% Only plot for a small number of representative age levels:
nPlot = 3;                          % How many age levels to pick: 3.
dt = (T-1)/(nPlot-1);               % Pick ages evenly spaced, start at t=1
tPlot = round(1+ ( 0:(nPlot-1) )*dt);% and end at T.
figure;
for it=1:nPlot                      % Loop over all ages.
    tt=tPlot(it);                   % Read out age.
    subplot(nPlot+1,1,it)           % Address the right subplot.
    plot(avec,c(:,:,tt));           % Plot consumption functions for all
                                    % income levels (at current age).
    title(sprintf('age: t=%1.0f',tt));% Title of subplot: age t.
    xlabel('a: assets')
    ylabel('c: consumption')   
end

% Example for how to calculate moments:
ctMean = sum(sum(c.*n,1),2);        % Sum over first and second dimension of
                                    % of c (weighing by the distribution)
                                    % to get mean consumption at each
                                    % age (recall that mass sums up to 1 at
                                    % each age): (1,1,T) array.
SSR = sum( sum(  n.*(c-ctMean).^2, 1), 2);  % Same for standard deviation:                               
ctStDev = sqrt( SSR );              % (1,1,T) array.
subplot(nPlot+1,1,nPlot+1)
plot(tvec(:),[ ctMean(:), ctStDev(:) ] ); % Plot mean and standard deviation
title('Consumption: moments by age')% of consumption. Note: 'ctMean(:)'
legend('mean','st.dev.')            % vectorizes ctMean and gives T-by-1
xlabel('t: age')                    % vector. 
ylabel('c: consumption')



