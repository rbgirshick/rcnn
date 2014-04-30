function res = rcnn_test(rcnn_model, imdb, suffix)
% res = rcnn_test(rcnn_model, imdb, suffix)
%   Compute test results using the trained rcnn_model on the
%   image database specified by imdb. Results are saved
%   with an optional suffix.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

conf = rcnn_config('sub_dir', imdb.name);
image_ids = imdb.image_ids;

% assume they are all the same
feat_opts = rcnn_model.training_opts;
num_classes = length(rcnn_model.classes);

if ~exist('suffix', 'var') || isempty(suffix)
  suffix = '';
else
  suffix = ['_' suffix];
end

try
  aboxes = cell(num_classes, 1);
  for i = 1:num_classes
    load([conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name suffix]);
    aboxes{i} = boxes;
  end
catch
  roidb = imdb.roidb_func(imdb);
  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    aboxes{i} = cell(length(image_ids), 1);
    box_inds{i} = cell(length(image_ids), 1);
  end

  % heuristic that yields at most 100k pre-NMS boxes
  % per 2500 images
  max_per_set = ceil(100000/2500)*length(image_ids);
  max_per_set = 30000;
  max_per_image = 100;
  top_scores = cell(num_classes, 1);
  thresh = -1.5*ones(num_classes, 1);
  box_counts = zeros(num_classes, 1);

  if ~isfield(rcnn_model, 'folds')
    folds{1} = 1:length(image_ids);
  else
    folds = rcnn_model.folds;
  end

  count = 0;
  for f = 1:length(folds)
    for i = folds{f}
      count = count + 1;
      fprintf('%s: test (%s) %d/%d\n', procid(), imdb.name, count, length(image_ids));
      th = tic;
      d = rcnn_load_cached_pool5_features(feat_opts.cache_name, ...
          imdb.name, image_ids{i});

      assert(size(d.feat, 1) == size(roidb.rois(i).boxes, 1));
      if isempty(d.feat)
        warning('empty image: %s', imdb_ids{i});
        continue;
      end
      
      d.feat = rcnn_pool5_to_fcX(d.feat, feat_opts.layer, rcnn_model);
      d.feat = rcnn_scale_features(d.feat, feat_opts.feat_norm_mean);
      zs = bsxfun(@plus, d.feat*rcnn_model.detectors(f).W, rcnn_model.detectors(f).B);
      fprintf('time 1: %.3fs\n', toc(th));

      th = tic;
      for j = 1:num_classes
        boxes = d.boxes;
        scores = zs(:,j);
        I = find(~d.gt & scores > thresh(j));
        keep = nms(cat(2, single(boxes(I,:)), single(scores(I))), 0.3);
        I = I(keep);
        if ~isempty(I)
          [~, ord] = sort(scores(I), 'descend');
          ord = ord(1:min(length(ord), max_per_image));
          I = I(ord);
          boxes = boxes(I,:);
          scores = scores(I);
          aboxes{j}{i} = cat(2, single(boxes), single(scores));
          box_inds{j}{i} = I;
        else
          aboxes{j}{i} = zeros(0, 5, 'single');
          box_inds{j}{i} = [];
        end

        if mod(count, 1000) == 0
          [aboxes{j}, box_inds{j}, thresh(j)] = ...
             keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
        end
      end
      fprintf('time 2: %.3fs\n', toc(th));
      if mod(count, 1000) == 0
        disp(thresh);
      end
    end
  end

  for i = 1:num_classes
    [aboxes{i}, box_inds{i}, thresh(i)] = ...
       keep_top_k(aboxes{i}, box_inds{i}, length(image_ids), ...
          max_per_set, thresh(i));

    save_file = [conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name suffix];
    boxes = aboxes{i};
    inds = box_inds{i};
    save(save_file, 'boxes', 'inds');
    clear boxes inds;
  end
end

% ------------------------------------------------------------------------
% Peform AP evaluation
% ------------------------------------------------------------------------
res = imdb.eval_func([], aboxes, imdb, suffix);


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
% Keep top K
X = cat(1, boxes{1:end_at});
if isempty(X)
  return;
end
scores = sort(X(:,end), 'descend');
thresh = scores(min(length(scores), top_k));
for image_index = 1:end_at
  bbox = boxes{image_index};
  keep = find(bbox(:,end) >= thresh);
  boxes{image_index} = bbox(keep,:);
  box_inds{image_index} = box_inds{image_index}(keep);
end
