function index = pool5_explorer_build_index(imdb, cache_name)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% index.imdb_name
% index.images
% index.features{i}.images_index
% index.features{i}.boxes
% index.features{i}.scores

% take a model
% select highest scoring features

conf = rcnn_config('sub_dir', imdb.name);

save_file = sprintf('%s/pool5_explorer_index_%s', ...
    conf.cache_dir, cache_name);

try
  % Load already computed index
  fprintf('trying to load: %s\n', save_file);
  load(save_file);
catch
  warning('Building the explorer index. This will take a long time.');

  TOP_K = 1000;

  ids = imdb.image_ids;

  % select features
  sel_features = 1:(6*6*256);

  index.imdb_name = imdb.name;
  index.images = ids;
  features = cell(length(sel_features), 1);

  for i = 1:length(features)
    features{i}.image_inds = [];
    features{i}.scores = [];
    features{i}.boxes = zeros(0, 4);
  end

  for i = 1:length(ids)
    tic_toc_print('%d/%d', i, length(ids));
    th = tic();
    d = rcnn_load_cached_pool5_features(cache_name, ...
        imdb.name, ids{i});

    feat = d.feat;

    parfor f = sel_features
      threshold = min(features{f}.scores);
      if isempty(threshold)
        threshold = -inf;
      end
      sel_0 = find(feat(:,f) > threshold);
      if isempty(sel_0)
        continue;
      end
      bs = [d.boxes(sel_0,:) feat(sel_0,f)];
      sel = nms(bs, 0.1);

      sel = sel_0(sel);
      sz = length(sel);

      new_image_inds = i*ones(sz,1);
      new_scores = feat(sel,f);
      new_boxes = d.boxes(sel,:);

      features{f}.image_inds = cat(1, features{f}.image_inds, ...
                                      new_image_inds);
      features{f}.scores = cat(1, features{f}.scores, ...
                                  new_scores);
      features{f}.boxes = cat(1, features{f}.boxes, ...
                                 new_boxes);

      [~, ord] = sort(features{f}.scores, 'descend');
      if length(ord) > TOP_K 
        ord = ord(1:TOP_K);
      end
      features{f}.image_inds = features{f}.image_inds(ord);
      features{f}.scores = features{f}.scores(ord);
      features{f}.boxes = features{f}.boxes(ord, :);
    end
    fprintf(' %.3fs\n', toc(th));

    if mod(i, 50) == 0
      index.features = features;
      save(save_file, 'index');
      fprintf('checkpoint %d\n', i);
    end
  end

  index.features = features;
  save(save_file, 'index');
end
