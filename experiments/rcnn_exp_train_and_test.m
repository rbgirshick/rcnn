function [res_test, res_train] = rcnn_exp_train_and_test()
% Runs an experiment that trains an R-CNN model and tests it.

% -------------------- CONFIG --------------------
net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
crop_mode    = 'warp';
crop_padding = 16;
layer        = 7;
k_folds      = 0;

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------

imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
imdb_test = imdb_from_voc(VOCdevkit, 'test', '2007');

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
