function [images, boxes] = selective_search_boxes_batch(imdb)

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
