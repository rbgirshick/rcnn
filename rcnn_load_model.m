function rcnn_model = rcnn_load_model(rcnn_model_or_file, use_gpu)

if isstr(rcnn_model_or_file)
  assert(exist(rcnn_model_or_file, 'file') ~= 0);
  ld = load(rcnn_model_or_file);
  rcnn_model = ld.rcnn_model; clear ld;
else
  rcnn_model = rcnn_model_or_file;
end

rcnn_model.cnn.init_key = ...
    caffe('init', rcnn_model.cnn.definition_file, rcnn_model.cnn.binary_file);
if exist('use_gpu', 'var') && ~use_gpu
  caffe('set_mode_cpu');
else
  caffe('set_mode_gpu');
end
caffe('set_phase_test');
rcnn_model.cnn.layers = caffe('get_weights');
