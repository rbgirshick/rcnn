function rcnn_model = fc8_model(rcnn_model)

opts.net_def_file = './model-defs/pascal_finetune_deploy.prototxt';
opts.net_file = './external/caffe/snapshots/v1/finetune_voc_2007_trainval_iter_70000';

rcnn_model.cnn.init_key = ...
    caffe('init', opts.net_def_file, opts.net_file);
caffe('set_phase_test');
rcnn_model.cnn.layers = caffe('get_weights');

rcnn_model.detectors.W = rcnn_model.cnn.layers(8).weights{1}(:,2:end);
rcnn_model.detectors.B = rcnn_model.cnn.layers(8).weights{2}(2:end)';
