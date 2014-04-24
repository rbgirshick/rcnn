function res = rcnn_exp_bbox_reg_train_and_test()
% Runs an experiment that trains a bounding box regressor and
% tests it.

% change to point to your VOCdevkit install
devkit = './datasets/ILSVRC13';

imdb_val1 = imdb_from_ilsvrc13(devkit, 'val1');
imdb_val2 = imdb_from_ilsvrc13(devkit, 'val2');

% load the rcnn_model trained by rcnn_exp_train_and_test()
conf = rcnn_config('sub_dir', imdb_val1.name);
ld = load([conf.cache_dir 'rcnn_model']);

% train the bbox regression model
bbox_reg = rcnn_train_bbox_regressor(imdb_val1, ld.rcnn_model, ...
    'min_overlap', 0.6, ...
    'layer', 5, ...
    'lambda', 1000, ...
    'robust', 0, ...
    'binarize', false);

% test the bbox regression model
res = rcnn_test_bbox_regressor(imdb_val2, ld.rcnn_model, bbox_reg, 'bbox_reg');
