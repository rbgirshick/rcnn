function vis_debug_bbox_pred(imdb, rcnn_model, bbox_reg, cls)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

image_ids = imdb.image_ids;
min_overlap = 0.3;
feat_opts = bbox_reg.training_opts;

roidb = imdb.roidb_func(imdb);
class_id = imdb.class_to_id(cls);

bbox_regressor = bbox_reg.models{class_id};

perm1 = randperm(length(image_ids));
%for i = 1:length(image_ids)
for i = perm1
  fprintf('%s: debug (%s) %d/%d\n', procid(), imdb.name, i, length(image_ids));

  if exist('cls','var')
    if isempty(find(roidb.rois(i).class == class_id))
      continue;
    end
  end

  d = rcnn_load_cached_pool5_features(feat_opts.cache_name, ...
      imdb.name, image_ids{i});
  if isempty(d.feat)
    continue;
  end

  d.feat = rcnn_pool5_to_fcX(d.feat, feat_opts.layer, rcnn_model);
  d.feat = rcnn_scale_features(d.feat, feat_opts.feat_norm_mean);

  if feat_opts.binarize
    d.feat = (d.feat > 0);
  end

  im = imread(imdb.image_at(i));

  if exist('cls','var')
    sel_gt = find(d.class == class_id);
  else
    sel_gt = find(d.class > 0);
  end
  gt_boxes = d.boxes(sel_gt, :);
  gt_classes = d.class(sel_gt);

  if exist('cls','var')
    max_ov = d.overlap(:, class_id);
  else
    max_ov = max(d.overlap, [], 2);
  end
  sel_ex = find(max_ov >= min_overlap);
  ex_boxes = d.boxes(sel_ex, :);
  ex_feat = d.feat(sel_ex, :);

  %for j = 1:size(ex_boxes, 1)
  perm = randperm(size(ex_boxes,1), min(10, size(ex_boxes,1)));
  for j = perm
    ex_box = ex_boxes(j, :);
    ov = boxoverlap(gt_boxes, ex_box);
    [max_ov, assignment] = max(ov);
    gt_box = gt_boxes(assignment, :);
    cls = gt_classes(assignment);

    src_w = ex_box(3) - ex_box(1) + eps;
    src_h = ex_box(4) - ex_box(2) + eps;
    src_ctr_x = ex_box(1) + 0.5*src_w;
    src_ctr_y = ex_box(2) + 0.5*src_h;
    
    gt_w = gt_box(3) - gt_box(1) + eps;
    gt_h = gt_box(4) - gt_box(2) + eps;
    gt_ctr_x = gt_box(1) + 0.5*gt_w;
    gt_ctr_y = gt_box(2) + 0.5*gt_h;

%    dst_ctr_x = (gt_ctr_x - src_ctr_x) * 1/src_w;
%    dst_ctr_y = (gt_ctr_y - src_ctr_y) * 1/src_h;
%    dst_scl_w = log(gt_w / src_w);
%    dst_scl_h = log(gt_h / src_h);
%
%    target = [dst_ctr_x dst_ctr_y dst_scl_w dst_scl_h];

    pred_box = rcnn_predict_bbox_regressor(bbox_regressor, ex_feat(j,:), ex_box);

    pred_box(1) = max(pred_box(1), 1);
    pred_box(2) = max(pred_box(2), 1);
    pred_box(3) = min(pred_box(3), size(im,2));
    pred_box(4) = min(pred_box(4), size(im,1));

    ovs = boxoverlap(cat(1, pred_box, ex_box), gt_box);

    showboxesc(im, gt_box, 'g', '-');
    showboxesc([], ex_box, 'b', '-');
    showboxesc([], pred_box, 'r', '-');
    hold on;
    plot(gt_ctr_x, gt_ctr_y, 'gd');
    plot(src_ctr_x, src_ctr_y, 'rd');
    hold off;
    title(sprintf('green = GT box;  blue = original box;  red = predicted box;  orig %.3f  pred %.3f', ovs(2), ovs(1)));

    pause;
  end
end
