function [images, boxes] = selective_search_boxes_batch(imdb)
% [images, boxes] = selective_search_boxes_batch(imdb)
%
% This function depends on simple-cluster-lib, which is specific to 
% the Berkeley cluster and not useful to the general public (and 
% hence not available). This file exists because it's convenient for 
% me to keep it in the repository.
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

fast_mode = true;

% ------------------------------------------------------------------------
% distributed computing configurations
addpath(genpath('rootdir/simple-cluster-lib'));
dconf                 = simple_cluster_lib_config();
dconf.cd              = pwd();
dconf.local           = false;
dconf.dist_nodes      = 20;
dconf.hours           = 12;
dconf.cput            = 60*60*24;
dconf.cleanup         = false;
%dconf.resume = true;
%dconf.work_dir_suffix = [testset '_' year];
% ------------------------------------------------------------------------

boxes = distributed_job(dconf, 'op_selective_search_boxes', ...
                        1, length(imdb.image_ids), imdb);
% mimic selective search output variable names
images = imdb.image_ids;
