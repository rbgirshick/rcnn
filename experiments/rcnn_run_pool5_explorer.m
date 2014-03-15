function rcnn_run_pool5_explorer()

% Example of how to use RCNN_CONFIG_OVERRIDE in a function
%global RCNN_CONFIG_OVERRIDE;
%conf_override.exp_dir = './cachedir/rcnn/v1.1/pool5';
%RCNN_CONFIG_OVERRIDE = @() conf_override;

% -------------------- CONFIG --------------------
cache_name = 'v1_finetune_voc_2007_trainval_iter_70k';

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------

imdb = imdb_from_voc(VOCdevkit, 'test', '2007');

pool5_explorer(imdb, cache_name);

clear global RCNN_CONFIG_OVERRIDE;
