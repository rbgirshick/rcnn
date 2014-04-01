function receptive_field_sizes()

% compute input size from a given output size
f = @(output_size, ksize, stride) (output_size - 1) * stride + ksize;

% fix the pool5 output size to 1 and derive the receptive field in the input
out = ...
f(f(f(f(f(f(f(f(1, 3, 2), ...   % conv5 -> pool5
                   3, 1), ...   % conv4 -> conv5
                   3, 1), ...   % conv3 -> conv4
                   3, 1), ...   % pool2 -> conv3
                   3, 2), ...   % conv2 -> pool2
                   5, 1), ...   % pool1 -> conv2
                   3, 2), ...   % conv1 -> pool1
                   11, 4);      % input -> conv1

fprintf('pool5 receptive field size: %d\n', out);

out = ...
f(f(f(f(f(f(f(1, 3, 1), ...   % conv4 -> conv5
                 3, 1), ...   % conv3 -> conv4
                 3, 1), ...   % pool2 -> conv3
                 3, 2), ...   % conv2 -> pool2
                 5, 1), ...   % pool1 -> conv2
                 3, 2), ...   % conv1 -> pool1
                 11, 4);      % input -> conv1

fprintf('conv5 receptive field size: %d\n', out);

out = ...
f(f(f(f(f(f(1, 3, 1), ...   % conv3 -> conv4
               3, 1), ...   % pool2 -> conv3
               3, 2), ...   % conv2 -> pool2
               5, 1), ...   % pool1 -> conv2
               3, 2), ...   % conv1 -> pool1
               11, 4);      % input -> conv1

fprintf('conv4 receptive field size: %d\n', out);

out = ...
f(f(f(f(f(1, 3, 1), ...   % pool2 -> conv3
             3, 2), ...   % conv2 -> pool2
             5, 1), ...   % pool1 -> conv2
             3, 2), ...   % conv1 -> pool1
             11, 4);      % input -> conv1

fprintf('conv3 receptive field size: %d\n', out);

out = ...
f(f(f(f(1, 3, 2), ...   % conv2 -> pool2
           5, 1), ...   % pool1 -> conv2
           3, 2), ...   % conv1 -> pool1
           11, 4);      % input -> conv1

fprintf('pool2 receptive field size: %d\n', out);

out = ...
f(f(f(1, 5, 1), ...   % pool1 -> conv2
         3, 2), ...   % conv1 -> pool1
         11, 4);      % input -> conv1

fprintf('conv2 receptive field size: %d\n', out);

out = ...
f(f(1, 3, 2), ...   % conv1 -> pool1
       11, 4);      % input -> conv1

fprintf('pool1 receptive field size: %d\n', out);

out = ...
f(1, 11, 4);      % input -> conv1

fprintf('conv1 receptive field size: %d\n', out);
