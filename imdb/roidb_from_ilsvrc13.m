function roidb = roidb_from_ilsvrc13(imdb)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

cache_file = ['./imdb/cache/roidb_' imdb.name];
try
  load(cache_file);
catch
  addpath(fullfile(imdb.details.devkit_path, 'evaluation')); 

  roidb.name = imdb.name;

  fprintf('Loading region proposals...');
  regions_file = sprintf('./data/selective_search_data/%s', roidb.name);
  regions = load(regions_file);
  fprintf('done\n');

  is_train = false;
  match = regexp(roidb.name, 'ilsvrc13_train_pos_(?<class_num>\d+)', 'names');
  if ~isempty(match)
    is_train = true;
  elseif ~strcmp(roidb.name, 'ilsvrc13_val')
    error('unknown image set');
  end

  hash = make_hash(imdb.details.meta_det.synsets);

  for i = 1:length(imdb.image_ids)
    tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
    if is_train
      anno_file = fullfile(imdb.details.bbox_path.train, ...
          get_wnid(imdb.image_ids{i}), [imdb.image_ids{i} '.xml']);
    else
      anno_file = fullfile(imdb.details.bbox_path.val, ...
          [imdb.image_ids{i} '.xml']);
    end

    try
      rec = VOCreadrecxml(anno_file, hash);
    catch
      rec = [];
    end
    roidb.rois(i) = attach_proposals(rec, regions.boxes{i});
  end

  rmpath(fullfile(imdb.details.devkit_path, 'evaluation')); 

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals(ilsvrc_rec, boxes)
% ------------------------------------------------------------------------

num_classes = 200;

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(ilsvrc_rec, 'objects') && length(ilsvrc_rec.objects) > 0
  gt_boxes = cat(1, ilsvrc_rec.objects(:).bbox);
  all_boxes = cat(1, gt_boxes, boxes);
  gt_classes = cat(1, ilsvrc_rec.objects(:).label);
  num_gt_boxes = size(gt_boxes, 1);
else
  gt_boxes = [];
  all_boxes = boxes;
  gt_classes = [];
  num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.is_difficult = false(num_gt_boxes + num_boxes, 1);
rec.overlap = zeros(num_gt_boxes+num_boxes, num_classes);
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.overlap = sparse(rec.overlap);
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
