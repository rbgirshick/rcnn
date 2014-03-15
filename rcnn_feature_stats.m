function [mean_norm, stdd] = rcnn_feature_stats(imdb, layer, rcnn_model)
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
save_file = sprintf('%s/feature_stats_%s_layer_%d_%s.mat', ...
                    conf.cache_dir, imdb.name, layer, rcnn_model.cache_name);

try
  ld = load(save_file);
  mean_norm = ld.mean_norm;
  stdd = ld.stdd;
  clear ld;
catch
  % fix the random seed for repeatability
  prev_rng = seed_rand();

  image_ids = imdb.image_ids;

  num_images = min(length(image_ids), 200);
  boxes_per_image = 200;

  image_ids = image_ids(randperm(length(image_ids), num_images));

  ns = [];
  for i = 1:length(image_ids)
    tic_toc_print('feature stats: %d/%d\n', i, length(image_ids));

    d = rcnn_load_cached_pool5_features(rcnn_model.cache_name, ...
        imdb.name, image_ids{i});
    X = d.feat(randperm(size(d.feat,1), min(boxes_per_image, size(d.feat,1))), :);
    X = rcnn_pool5_to_fcX(X, layer, rcnn_model);

    ns = cat(1, ns, sqrt(sum(X.^2, 2)));
  end

  mean_norm = mean(ns);
  stdd = std(ns);
  save(save_file, 'mean_norm', 'stdd');

  % restore previous rng
  rng(prev_rng);
end
