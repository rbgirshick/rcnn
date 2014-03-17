function feat = rcnn_features(im, boxes, rcnn_model)
% feat = rcnn_features(im, boxes, rcnn_model)
%   Compute CNN features on a set of boxes.
%
%   im is an image in RGB order as returned by imread
%   boxes are in [x1 y1 x2 y2] format with one box per row
%   rcnn_model specifies the CNN Caffe net file to use.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% make sure that caffe has been initialized for this model
if rcnn_model.cnn.init_key ~= caffe('get_init_key')
  error('You probably need to call rcnn_load_model');
end

% Each batch contains 256 (default) image regions.
% Processing more than this many at once takes too much memory
% for a typical high-end GPU.
[batches, batch_padding] = rcnn_extract_regions(im, boxes, rcnn_model);
batch_size = rcnn_model.cnn.batch_size;

% compute features for each batch of region images
feat_dim = -1;
feat = [];
curr = 1;
for j = 1:length(batches)
  % forward propagate batch of region images 
  f = caffe('forward', batches(j));
  f = f{1};
  
  % first batch, init feat_dim and feat
  if j == 1
    f_size = size(f);
    assert(f_size(end) == batch_size);
    feat_dim = prod(f_size(1:end-1));
    feat = zeros(size(boxes, 1), feat_dim, 'single');
  end

  f = reshape(f, [feat_dim batch_size]);

  % last batch, trim f to size
  if j == length(batches)
    if batch_padding > 0
      f = f(:, 1:end-batch_padding);
    end
  end

  feat(curr:curr+size(f,2)-1,:) = f';
  curr = curr + batch_size;
end
