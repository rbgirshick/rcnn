function rcnn_model = to_rcnn_model()

load cachedir/convnet-selective-search/caffe2_warp_context16_ft_context16_lr10x/trainval_2007_vs_test_2007_layer_fc6_finetuned_1_repro/pos_gt_neg_0.3_svm_C_0.00100_w1_2/2007/rcnn_model.mat;
rcnn_model.detectors.training_opts.cache_name = 'voc_2007_finetuned_1_caffe2_warp_context16_lr10x';
rcnn_model.cache_name = 'voc_2007_finetuned_1_caffe2_warp_context16_lr10x';
load cachedir/convnet-selective-search/caffe2_warp_context16_ft_context16_lr10x/trainval_2007_vs_test_2007_layer_fc7_finetuned_1/pos_gt_neg_0.3_svm_C_0.00100_w1_2/2007/models_final.mat;

rcnn_model.detectors.training_opts.feat_norm_mean = models{1}.opts.feat_norm_mean;
W = cat(2, cellfun(@(x) x.w, models, 'UniformOutput', false));
W = cat(2, W{:});
B = cat(2, cellfun(@(x) x.b, models, 'UniformOutput', false));
B = cat(2, B{:});
rcnn_model.detectors.W = W;
rcnn_model.detectors.B = B;
rcnn_model.detectors.training_opts.layer = 7;
