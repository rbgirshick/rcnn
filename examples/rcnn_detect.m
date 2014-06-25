function dets = rcnn_detect(im, rcnn_model, thresh)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% compute selective search candidates
fprintf('Computing candidate regions...');
th = tic();
fast_mode = true;
im_width = 500;
boxes = selective_search_boxes(im, fast_mode, im_width);
% compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);
fprintf('found %d candidates (in %.3fs).\n', size(boxes,1), toc(th));

% extract features from candidates (one row per candidate box)
fprintf('Extracting CNN features from regions...');
th = tic();
feat = rcnn_features(im, boxes, rcnn_model);
feat = rcnn_scale_features(feat, rcnn_model.training_opts.feat_norm_mean);
fprintf('done (in %.3fs).\n', toc(th));

% compute scores for each candidate [num_boxes x num_classes]
fprintf('Scoring regions with detectors...');
th = tic();
scores = bsxfun(@plus, feat*rcnn_model.detectors.W, rcnn_model.detectors.B);
fprintf('done (in %.3fs)\n', toc(th));

% apply NMS to each class and return final scored detections
fprintf('Applying NMS...');
th = tic();
num_classes = length(rcnn_model.classes);
dets = cell(num_classes, 1);
for i = 1:num_classes
  I = find(scores(:, i) > thresh);
  scored_boxes = cat(2, boxes(I, :), scores(I, i));
  keep = nms(scored_boxes, 0.3); 
  dets{i} = scored_boxes(keep, :);
end
fprintf('done (in %.3fs)\n', toc(th));
