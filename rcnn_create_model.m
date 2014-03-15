function rcnn_model = rcnn_create_model(cnn_definition_file, cnn_binary_file, cache_name)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

if ~exist('cache_name', 'var') || isempty(cache_name)
  cache_name = 'none';
end

%  model = 
%    cnn: [1x1 struct]
%        binary_file: 'path/to/cnn/model/binary'
%        definition_file: 'path/to/cnn/model/definition'
%        batch_size: 256
%        image_mean: [227x227x3 single]
%        init_key: -1
%    detectors.W: [N x <numclasses> single]  % matrix of SVM weights
%    detectors.B: [1 x <numclasses> single]  % (row) vector of SVM biases
%    detectors.crop_mode: 'warp' or 'square'
%    detectors.crop_padding: 16
%    detectors.nms_thresholds: [1x20 single]
%    training_opts: [1x1 struct]
%        bias_mult: 10
%        fine_tuned: 1
%        layer: 'fc7'
%        pos_loss_weight: 2
%        svm_C: 1.0000e-03
%        trainset: 'trainval'
%        use_flipped: 0
%        year: '2007'
%        feat_norm_mean: 20.1401
%    classes: {cell array of class names}
%    class_to_index: map from class name to column index in W

% init empty convnet
assert(exist(cnn_binary_file, 'file') ~= 0);
assert(exist(cnn_definition_file, 'file') ~= 0);
cnn.binary_file = cnn_binary_file;
cnn.definition_file = cnn_definition_file;
cnn.batch_size = 256;
cnn.init_key = -1;
cnn.input_size = 227;
% load the ilsvrc image mean
data_mean_file = './external/caffe/matlab/caffe/ilsvrc_2012_mean.mat';
assert(exist(data_mean_file, 'file') ~= 0);
ld = load(data_mean_file);
image_mean = ld.image_mean; clear ld;
off = floor((size(image_mean,1) - cnn.input_size)/2)+1;
image_mean = image_mean(off:off+cnn.input_size-1, off:off+cnn.input_size-1, :);
cnn.image_mean = image_mean;

% init empty detectors
detectors.W = [];
detectors.B = [];
detectors.crop_mode = 'warp';
detectors.crop_padding = 16;
detectors.nms_thresholds = [];

% rcnn model wraps the convnet and detectors
rcnn_model.cnn = cnn;
rcnn_model.cache_name = cache_name;
rcnn_model.detectors = detectors;
