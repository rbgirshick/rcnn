%
% Net fine-tuned on voc 2012 train
%
net_file     = './data/caffe_nets/finetune_voc_2012_train_iter_70k';
cache_name   = 'v1_finetune_voc_2012_train_iter_70k';
crop_mode    = 'warp';
crop_padding = 16;

%
% Net fine-tuned on voc 2007 trainval
%
net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
crop_mode    = 'warp';
crop_padding = 16;

%
% Net pre-trained (no fine-tuning) on ILSVRC 2012 train with warped examples
%
net_file     = './data/caffe_nets/ilsvrc_2012_train_iter_310k';
cache_name   = 'v1_caffe_imagenet_train_iter_310k';
crop_mode    = 'warp';
crop_padding = 16;
