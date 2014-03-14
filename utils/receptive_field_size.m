function out = receptive_field_size()

% conv1  11   55x55
% conv2  27   55x55
% pool2  35   27x27
% conv3  51   27x27  
% pool3  67   13x13
% conv4  99   13x13
% conv5 131   13x13
% pool5 163   6x6

out = ...
  pool3_to_conv3(...
  conv4_to_pool3(...
  conv5_to_conv4(...
  pool5_to_conv5(1))));

return

out = ...
  conv1_to_input(...
  conv2_to_conv1(...
  pool2_to_conv2(...
  conv3_to_pool2(...
  pool3_to_conv3(...
  conv4_to_pool3(...
  conv5_to_conv4(...
  pool5_to_conv5(1))))))));

function out = pool5_to_conv5(p)
out = 2*(p-1)+1 + 2*floor(3/2);

function out = conv5_to_conv4(p)
out = 1*(p-1)+1 + 2*floor(3/2);

function out = conv4_to_pool3(p)
out = 1*(p-1)+1 + 2*floor(3/2);

function out = pool3_to_conv3(p)
out = 2*(p-1)+1 + 2*floor(3/2);

function out = conv3_to_pool2(p)
out = 1*(p-1)+1 + 2*floor(3/2);

function out = pool2_to_conv2(p)
out = 2*(p-1)+1 + 2*floor(3/2);

function out = conv2_to_conv1(p)
out = 1*(p-1)+1 + 2*floor(5/2);

function out = conv1_to_input(p)
out = 4*(p-1)+1 + 2*floor(11/2);
