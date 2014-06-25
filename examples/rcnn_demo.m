function rcnn_demo(demo_choice, use_gpu)
% rcnn_demo(demo_choice, use_gpu)
%   Run the R-CNN demo on a test image. Set use_gpu = false to run
%   in CPU mode. (GPU mode is the default.)
%   demo_choice selects between fine-tuned R-CNN models trained on 
%   'PASCAL' or 'ILSVRC13' 

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

clf;

thresh = -1;

if ~exist('demo_choice', 'var') || isempty(demo_choice)
  demo_choice = 'PASCAL';
end

switch demo_choice
  case 'PASCAL'
    % Example using the PASCAL VOC 2007 fine-tuned detectors (20 classes)
    rcnn_model_file = './data/rcnn_models/voc_2012/rcnn_model_finetuned.mat';
    im = imread('./examples/images/000084.jpg');
  case 'ILSVRC13'
    % Example using the ILSVRC13 fine-tuned detectors (200 classes)
    rcnn_model_file = './data/rcnn_models/ilsvrc2013/rcnn_model.mat';
    im = imread('./examples/images/fish-bike.jpg');
  otherwise
    error('unknown demo ''%s'' [valid options: ''PASCAL'' or ''ILSVRC13'']', demo_choice);
end

if ~exist(rcnn_model_file, 'file')
  error('You need to download the R-CNN precomputed models. See README.md for details.');
end

if ~exist('use_gpu', 'var') || isempty(use_gpu)
  use_gpu = true;
end

modes = {'CPU', 'GPU'};
fprintf('~~~~~~~~~~~~~~~~~~~\n');
fprintf('Welcome to the %s demo\n', demo_choice);
fprintf('Running in %s mode\n', modes{use_gpu+1});
fprintf('(To run in %s mode, call rcnn_demo(demo_choice, %d) instead)\n',  ...
    modes{~use_gpu+1}, ~use_gpu);
fprintf('Press any key to continue\n');
pause;

% Initialization only needs to happen once (so this time isn't counted
% when timing detection).
fprintf('Initializing R-CNN model (this might take a little while)\n');
rcnn_model = rcnn_load_model(rcnn_model_file, use_gpu);
fprintf('done\n');

th = tic;
dets = rcnn_detect(im, rcnn_model, thresh);
fprintf('Total %d-class detection time: %.3fs\n', ...
    length(rcnn_model.classes), toc(th));

all_dets = [];
for i = 1:length(dets)
  all_dets = cat(1, all_dets, ...
      [i * ones(size(dets{i}, 1), 1) dets{i}]);
end

[~, ord] = sort(all_dets(:,end), 'descend');
for i = 1:length(ord)
  score = all_dets(ord(i), end);
  if score < 0
    break;
  end
  cls = rcnn_model.classes{all_dets(ord(i), 1)};
  showboxes(im, all_dets(ord(i), 2:5));
  title(sprintf('det #%d: %s score = %.3f', ...
      i, cls, score));
  drawnow;
  pause;
end

fprintf('No more detection with score >= 0\n');
