function res = rcnn_test_bbox_regressor(imdb, rcnn_model, bbox_reg, suffix)
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
feat_opts = bbox_reg.training_opts;
num_classes = length(rcnn_model.classes);

if ~exist('suffix', 'var') || isempty(suffix)
  suffix = '_bbox_reg';
else
  if suffix(1) ~= '_'
    suffix = ['_' suffix];
  end
end

try
  aboxes = cell(num_classes, 1);
  for i = 1:num_classes
    load([conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name suffix]);
    aboxes{i} = boxes;
  end
catch
  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    load([conf.cache_dir rcnn_model.classes{i} '_boxes_' imdb.name]);
    aboxes{i} = boxes;
    box_inds{i} = inds;
    clear boxes inds;
  end

  for i = 1:length(image_ids)
    fprintf('%s: bbox reg test (%s) %d/%d\n', procid(), imdb.name, i, length(image_ids));
    d = rcnn_load_cached_pool5_features(feat_opts.cache_name, ...
        imdb.name, image_ids{i});
    if isempty(d.feat)
      continue;
    end

    d.feat = rcnn_pool5_to_fcX(d.feat, feat_opts.layer, rcnn_model);
    d.feat = rcnn_scale_features(d.feat, feat_opts.feat_norm_mean);

    if feat_opts.binarize
      d.feat = single(d.feat > 0);
    end

    for j = 1:num_classes
      I = box_inds{j}{i};
      boxes = aboxes{j}{i};
      if ~isempty(boxes)
        scores = boxes(:,end);
        boxes = boxes(:,1:4);
        assert(sum(sum(abs(d.boxes(I,:) - boxes))) == 0);
        boxes = rcnn_predict_bbox_regressor(bbox_reg.models{j}, d.feat(I,:), boxes);
        boxes(:,1) = max(boxes(:,1), 1);
        boxes(:,2) = max(boxes(:,2), 1);
        boxes(:,3) = min(boxes(:,3), imdb.sizes(i,2));
        boxes(:,4) = min(boxes(:,4), imdb.sizes(i,1));
        aboxes{j}{i} = cat(2, single(boxes), single(scores));

        if 0
          % debugging visualizations
          im = imread(imdb.image_at(i));
          keep = nms(aboxes{j}{i}, 0.3);
          for k = 1:min(10, length(keep))
            if aboxes{j}{i}(keep(k),end) > -0.9
              showboxes(im, aboxes{j}{i}(keep(k),1:4));
              title(sprintf('%s %d score: %.3f\n', rcnn_model.classes{j}, ...
                  k, aboxes{j}{i}(keep(k),end)));
              pause;
            end
          end
        end
      end
    end
  end

  for i = 1:num_classes
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
for model_ind = 1:num_classes
  cls = rcnn_model.classes{model_ind};
  try
    ld = load([conf.cache_dir cls '_pr_' imdb.name suffix]);
    fprintf('!!! %s : %.4f %.4f\n', cls, ld.res.ap, ld.res.ap_auc);
    res(model_ind) = ld.res;
  catch
    res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, suffix);
  end
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results (bbox reg):\n');
aps = [res(:).ap]';
disp(aps);
disp(mean(aps));
fprintf('~~~~~~~~~~~~~~~~~~~~\n');
