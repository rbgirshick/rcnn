function bbox_reg = rcnn_train_bbox_regressor(imdb, rcnn_model, varargin)
% bbox_reg = rcnn_train_bbox_regressor(imdb, rcnn_model, varargin)
%   Trains a bounding box regressor on the image database imdb
%   for use with the R-CNN model rcnn_model. The regressor is trained
%   using ridge regression.
%
%   Keys that can be passed in:
%
%   min_overlap     Proposal boxes with this much overlap or more are used
%   layer           The CNN layer features to regress from (either 5, 6 or 7)
%   lambda          The regularization hyperparameter in ridge regression
%   robust          Throw away examples with loss in the top [robust]-quantile
%   binarize        Binarize features or leave as real values >= 0

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addRequired('imdb',       @isstruct);
ip.addRequired('rcnn_model', @isstruct);
ip.addParamValue('min_overlap', 0.6,   @isscalar);
ip.addParamValue('layer',       5,     @isscalar);
ip.addParamValue('lambda',      1000,  @isscalar);
ip.addParamValue('robust',      0,     @isscalar);
ip.addParamValue('binarize',    false, @islogical);

ip.parse(imdb, rcnn_model, varargin{:});
opts = ip.Results;
opts = rmfield(opts, 'rcnn_model');
opts = rmfield(opts, 'imdb');
opts.cache_name = rcnn_model.cache_name;

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

conf = rcnn_config('sub_dir', imdb.name);
clss = rcnn_model.classes;
num_clss = length(clss);

% ------------------------------------------------------------------------
% Get the average norm of the features
opts.feat_norm_mean = rcnn_feature_stats(imdb, opts.layer, rcnn_model);
fprintf('average norm = %.3f\n', opts.feat_norm_mean);
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get all positive examples
save_file = sprintf('./feat_cache/%s/%s/bbox_regressor_XY_layer_5_overlap_0.5.mat', ...
                    rcnn_model.cache_name, imdb.name);
try
  load(save_file);
  fprintf('Loaded saved positives from ground truth boxes\n');
catch
  [X, Y, O, C] = get_examples(rcnn_model, imdb, opts);
  save(save_file, 'X', 'Y', 'O', 'C', '-v7.3');
end
for i = 1:num_clss
  fprintf('%14s has %6d samples\n', rcnn_model.classes{i}, length(find(C == i)));
end
X = rcnn_pool5_to_fcX(X, opts.layer, rcnn_model);
X = rcnn_scale_features(X, opts.feat_norm_mean);
% ------------------------------------------------------------------------

% use ridge regression solved by cholesky factorization
method = 'ridge_reg_chol';

models = cell(num_clss, 1);
for i = 1:num_clss
  fprintf('Training regressors for class %s (%d/%d)\n', ...
      rcnn_model.classes{i}, i, num_clss);
  I = find(O > opts.min_overlap & C == i);
  Xi = X(I,:); 
  if opts.binarize
    Xi = single(Xi > 0);
  end
  Yi = Y(I,:); 
  Oi = O(I); 
  Ci = C(I);

  % add bias feature
  Xi = cat(2, Xi, ones(size(Xi,1), 1, class(Xi)));

  % Center and decorrelate targets
  mu = mean(Yi);
  Yi = bsxfun(@minus, Yi, mu);
  S = Yi'*Yi / size(Yi,1);
  [V, D] = eig(S);
  D = diag(D);
  T = V*diag(1./sqrt(D+0.001))*V';
  T_inv = V*diag(sqrt(D+0.001))*V';
  Yi = Yi * T;

  models{i}.mu = mu;
  models{i}.T = T;
  models{i}.T_inv = T_inv;

  models{i}.Beta = [ ...
    solve_robust(Xi, Yi(:,1), opts.lambda, method, opts.robust) ...
    solve_robust(Xi, Yi(:,2), opts.lambda, method, opts.robust) ...
    solve_robust(Xi, Yi(:,3), opts.lambda, method, opts.robust) ...
    solve_robust(Xi, Yi(:,4), opts.lambda, method, opts.robust)];
end
bbox_reg.models = models;
bbox_reg.training_opts = opts;
save([conf.cache_dir 'bbox_regressor_final'], 'bbox_reg');


% ------------------------------------------------------------------------
function [X, Y, O, C] = get_examples(rcnn_model, imdb, opts)
% ------------------------------------------------------------------------
num_classes = length(rcnn_model.classes);

pool5 = 5;

roidb = imdb.roidb_func(imdb);
cls_counts = zeros(num_classes, 1);
for i = 1:length(imdb.image_ids)
  tic_toc_print('%s: counting %d/%d\n', ...
                procid(), i, length(imdb.image_ids));

  d = roidb.rois(i);
  [max_ov cls] = max(d.overlap, [], 2);
  sel_ex = find(max_ov >= 0.5);
  cls = cls(sel_ex);
  for j = 1:length(sel_ex)
    cls_counts(cls(j)) = cls_counts(cls(j)) + 1;
  end
end
total = sum(cls_counts);
feat_dim = size(rcnn_model.cnn.layers(pool5+1).weights{1},1);
% features
X = zeros(total, feat_dim, 'single');
% target values
Y = zeros(total, 4, 'single');
% overlap amounts
O = zeros(total, 1, 'single');
% classes
C = zeros(total, 1, 'single');
cur = 1;

for i = 1:length(imdb.image_ids)
  tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(imdb.image_ids));

  d = rcnn_load_cached_pool5_features(rcnn_model.cache_name, ...
      imdb.name, imdb.image_ids{i});

  sel_gt = find(d.class > 0);
  gt_boxes = d.boxes(sel_gt, :);
  gt_classes = d.class(sel_gt);

  max_ov = max(d.overlap, [], 2);
  sel_ex = find(max_ov >= opts.min_overlap);
  ex_boxes = d.boxes(sel_ex, :);

  X(cur+(0:length(sel_ex)-1), :) = d.feat(sel_ex, :);

  for j = 1:size(ex_boxes, 1)
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

    dst_ctr_x = (gt_ctr_x - src_ctr_x) * 1/src_w;
    dst_ctr_y = (gt_ctr_y - src_ctr_y) * 1/src_h;
    dst_scl_w = log(gt_w / src_w);
    dst_scl_h = log(gt_h / src_h);

    target = [dst_ctr_x dst_ctr_y dst_scl_w dst_scl_h];

    if 0
      % debugging visualizations and checks
      im = imread(imdb.image_at(i));
      showboxesc(im, gt_box, 'g', '-');
      showboxesc([], ex_box, 'r', '-');
      hold on;
      plot(gt_ctr_x, gt_ctr_y, 'gd');
      plot(src_ctr_x, src_ctr_y, 'rd');
      hold off;
      fprintf('target = [%.3f %.3f %.3f %.3f]\n', target(1), target(2), target(3), target(4));
      fprintf('cls = %s\n', rcnn_model.classes{cls});

      % check that we can correctly reconstruct the gt_box from the
      % gold-standard target
      pred_ctr_x = (target(1) * src_w) + src_ctr_x;
      pred_ctr_y = (target(2) * src_h) + src_ctr_y;
      pred_w = exp(target(3)) * src_w;
      pred_h = exp(target(4)) * src_h;
      pred_box = [pred_ctr_x - 0.5*pred_w, pred_ctr_y - 0.5*pred_h, ...
                  pred_ctr_x + 0.5*pred_w, pred_ctr_y + 0.5*pred_h];
      disp(pred_box);
      disp(gt_box);
      assert(sum(abs(pred_box - gt_box)) < 0.0001);

      pause;
    end

    assert(cur <= total);
    Y(cur, :) = target;
    O(cur)    = max_ov;
    C(cur)    = cls;
    cur = cur + 1;
  end
end


% ------------------------------------------------------------------------
function [x, losses] = solve_robust(A, y, lambda, method, qtile)
% ------------------------------------------------------------------------
[x, losses] = solve(A, y, lambda, method);
fprintf('loss = %.3f\n', mean(losses));
if qtile > 0
  thresh = quantile(losses, 1-qtile);
  I = find(losses < thresh);
  [x, losses] = solve(A(I,:), y(I), lambda, method);
  fprintf('loss (robust) = %.3f\n', mean(losses));
end


% ------------------------------------------------------------------------
function [x, losses] = solve(A, y, lambda, method)
% ------------------------------------------------------------------------

%tic;
switch method
case 'ridge_reg_chol'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  %
  % solve (A'A + lambdaI)x = A'y for x using cholesky factorization
  % R'R = (A'A + lambdaI)
  % R'z = A'y  :  solve for z  =>  R'Rx = R'z  =>  Rx = z
  % Rx = z     :  solve for x
  R = chol(A'*A + lambda*eye(size(A,2)));
  z = R' \ (A'*y);
  x = R \ z;
case 'ridge_reg_inv'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  x = inv(A'*A + lambda*eye(size(A,2)))*A'*y;
case 'ls_mldivide'
  % solve for x in min_x ||Ax - y||^2
  if lambda > 0
    warning('ignoring lambda; no regularization used');
  end
  x = A\y;
end
%toc;
losses = 0.5 * (A*x - y).^2;
