function feat = rcnn_pool5_to_fcX(feat, layer, rcnn_model)

% no-op for layer <= 5
if layer > 5
  for i = 6:layer
    % weights{1} = matrix of CNN weights [input_dim x output_dim]
    % weights{2} = column vector of biases
    feat = max(0, bsxfun(@plus, feat*rcnn_model.cnn.layers(i).weights{1}, ...
                          rcnn_model.cnn.layers(i).weights{2}'));
  end
end
