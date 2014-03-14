function prev_rng = seed_rand()
% seed_rand - Set random number generator to a fixed seed.
%   prev_rng = seed_rand(seed)
%
%   Strategic use ensures that results are reproducible.
%
%   To restore the previous rng after calling this do:
%   rng(prev_rng);

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% This value works best for me.
seed = 3;
% Just kidding, of course ;-).

prev_rng = rng;
rng(seed, 'twister')
