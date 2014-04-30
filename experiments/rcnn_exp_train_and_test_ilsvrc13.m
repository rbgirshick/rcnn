function [res_test, res_train] = rcnn_exp_train_and_test_ilsvrc13()
% Runs an experiment that trains an R-CNN model and tests it.

% -------------------- CONFIG --------------------
%net_file     = './data/caffe_nets/ilsvrc_2012_train_iter_310k';
%cache_name   = 'v1_caffe_imagenet_train_iter_310k';
%crop_mode    = 'warp';
%crop_padding = 16;
%layer        = 6;
%k_folds      = 0;

%net_file     = '/data1/ILSVRC13/finetune_ilsvrc13_val1_iter_50000';
%cache_name   = 'v1_finetune_val1_iter_50k';
%crop_mode    = 'warp';
%crop_padding = 16;
%layer        = 7;
%k_folds      = 0;

net_file     = '/data1/ILSVRC13/finetune_ilsvrc13_val1+train1k_iter_50000';
cache_name   = 'v1_finetune_val1+train1k_iter_50k';
crop_mode    = 'warp';
crop_padding = 16;
layer        = 7;
k_folds      = 0;

% change to point to your VOCdevkit install
devkit = './datasets/ILSVRC13';
% ------------------------------------------------

%imdb_val1 = imdb_from_ilsvrc13(devkit, 'val1');
%imdb_val2 = imdb_from_ilsvrc13(devkit, 'val2');
%imdb_train = imdb_val1;
%imdb_test = imdb_val2;

imdb_val = imdb_from_ilsvrc13(devkit, 'val');
imdb_test = imdb_from_ilsvrc13(devkit, 'test');
imdb_train = imdb_val;


[rcnn_model, rcnn_k_fold_model] = ...
    rcnn_train(imdb_train, ...
      'layer',        layer, ...
      'k_folds',      k_folds, ...
      'cache_name',   cache_name, ...
      'net_file',     net_file, ...
      'crop_mode',    crop_mode, ...
      'crop_padding', crop_padding);

if k_folds > 0
  res_train = rcnn_test(rcnn_k_fold_model, imdb_train);
else
  res_train = [];
end

res_test = rcnn_test(rcnn_model, imdb_test);
