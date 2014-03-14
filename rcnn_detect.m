function dets = rcnn_detect(im, rcnn_model)

% compute selective search candidates
fprintf('Computing candidate regions...');
th = tic();
fast_mode = true;
boxes = selective_search_boxes(im, fast_mode);
% compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);
fprintf('found %d candidates (in %.3fs).\n', size(boxes,1), toc(th));

% extract features from candidates (one row per candidate box)
fprintf('Extracting CNN features from regions...');
th = tic();
feat = rcnn_features(im, boxes, rcnn_model);
feat = rcnn_scale_features(feat, rcnn_model.training_opts.feat_norm_mean);
fprintf('done (in %.3fs).\n', toc(th));

fprintf('Scoring regions with detectors and applying NMS...');
% compute scores for each candidate [num_boxes x num_classes]
th = tic();
scores = bsxfun(@plus, feat*rcnn_model.detectors.W, rcnn_model.detectors.B);

% apply NMS to each class and return final scored detections
num_classes = length(rcnn_model.classes);
dets = cell(num_classes, 1);
for i = 1:num_classes
  scored_boxes = cat(2, boxes, scores(:,i));
  keep = nms(scored_boxes, 0.3); 
  dets{i} = scored_boxes(keep, :);
end
fprintf('done (in %.3fs)\n', toc(th));
