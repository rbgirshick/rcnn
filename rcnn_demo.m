function rcnn_demo(use_gpu)
% rcnn_demo(use_gpu)
%   Run the R-CNN demo on a test image. Set use_gpu = false to run
%   in CPU mode. (GPU mode is the default.)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

rcnn_model_file = './data/rcnn_models/voc_2012/rcnn_model_finetuned.mat';

if ~exist(rcnn_model_file, 'file')
  error('You need to download the R-CNN precomputed models. See README.md for details.');
end

if ~exist('use_gpu', 'var') || isempty(use_gpu)
  use_gpu = true;
end

modes = {'CPU', 'GPU'};
fprintf('~~~~~~~~~~~~~~~~~~~\n');
fprintf('Running in %s mode\n', modes{use_gpu+1});
fprintf('(To run in %s mode, call rcnn_demo(%d) instead)\n',  ...
    modes{~use_gpu+1}, ~use_gpu);
fprintf('Press any key to continue\n');
pause;

fprintf('Initializing R-CNN model (this might take a little while)\n');
rcnn_model = rcnn_load_model(rcnn_model_file, use_gpu);
fprintf('done\n');

im = imread('./000084.jpg');

dets = rcnn_detect(im, rcnn_model);

% show top scoring bicycle detection
showboxes(im, dets{2}(1,:));
title(sprintf('bicycle conf = %.3f', dets{2}(1,end)));

fprintf('Press any key to see the top scoring person detection\n');
pause;

% show top scoring person detection
showboxes(im, dets{15}(1,:));
title(sprintf('person conf = %.3f', dets{15}(1,end)));
