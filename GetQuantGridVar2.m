function [qout,xbins] = GetQuantGridVar2(x,n,qtl)
% Returns the quantiles in qtl for a random variable x which is discretized
% on a grid x, which may be either linear or have increasing increments
% (e.g. housing 0-50-100-200-400-... in bequest example) with distribution 
% n. It is assumed that the mass n(i) is uniformly distributed over an interval
% (or "bin") centered around the grid point x(i). The bins are of the same
% size for linear grids, but of increasing size for grids with increasing
% increments.
%
% Inputs:
% x:        N-by-1 column vector of values for random variable. Either 
%           linearly spaced or having increasing increments.
% n:        N-by-1 column vector with distribution over these grid points.
%           Need not sum up to one. 
% qtl:      k-by-1 column vector with quantiles; optional 
%           (default is [0.1; 0.25; 0.5; 0.75; 0.9]).
% 
% Outputs:
% quant:    k-by-1 column vector with quantiles of x.
% xbins:    (N+1)-by-1 column vector with the bounds of the constructed bins.

N = numel(x);                   % Read out number of values.
if nargin<3                     % Default for qtl if not handed over:
    qtl = [0.1; 0.25; 0.5; 0.75; 0.9];
end

% First, set up bins for the grid. Each bin is centered around a grid
% point. Bin size is constant (=dx) for linearly-spaced grid, but bin sizes
% will be increasing if grid has increasing increments.
xbins = zeros(N+1,1);           % Set up bins for grid. 
dx = x(2)-x(1);                 % Get spacing of first two grid points.
xbins(1) = x(1)-dx/2;           % First bin starts left of x(1).
for i=1:N                       % Now, loop over all grid points.
    dxhalf     = x(i)-xbins(i); % Get distance of grid point to lower bin bound.
    xbins(i+1) = x(i) + dxhalf; % Add this distance on top to have bin
end                             % centered around x(i).

small = 10^(-10)/length(n);     % Set zero probabilities to very small value
n(n<small) = small;             % --> Need this for interpolation!            
nn = n/sum(n);                  % Normalize density.
NN = [ 0; cumsum(nn) ];         % Approximation for cdf evaluated at the bin
                                % end points. Assume here that density is 
                                % uniformly spread across the bin. By
                                % construction of bins, "mean agent" sits
                                % exactly on bin center and thus on the
                                % corresponding grid point in x.

% Interpolate the function mapping NN to xx at quantiles:
qout = interp1(NN,xbins,qtl);   % Now, just have to interpolate the cdf at 
                                % values of qtl to get the desired
                                % quantiles.
                              
 