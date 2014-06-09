function rcnn_model = fc8_model(rcnn_model)

opts.net_def_file = './model-defs/pascal_finetune_deploy.prototxt';

rcnn_model.cnn.init_key = ...
    caffe('init', opts.net_def_file, rcnn_model.cnn.binary_file);
caffe('set_phase_test');
rcnn_model.cnn.layers = caffe('get_weights');

rcnn_model.detectors.W = rcnn_model.cnn.layers(8).weights{1};
rcnn_model.detectors.B = rcnn_model.cnn.layers(8).weights{2}';
