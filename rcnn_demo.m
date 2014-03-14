function rcnn_demo()

rcnn_model_file = './data/rcnn_models/voc_2012/rcnn_model_finetuned.mat';

if ~exist(rcnn_model_file, 'file')
  error('You need to download the R-CNN precomputed models. See README.md for details.');
end

use_gpu = true;
fprintf('Initializing R-CNN model (this might take a little while)\n');
rcnn_model = rcnn_load_model(rcnn_model_file, use_gpu);
fprintf('done\n');

im = imread('./000084.jpg');

dets = rcnn_detect(im, rcnn_model);

% show top scoring bicycle detection
showboxes(im, dets{2}(1,:));
title(sprintf('bicycle conf = %.3f', dets{2}(1,end)));

fprintf('\n> Press any key to see the top scoring person detection.');
pause;

% show top scoring person detection
showboxes(im, dets{15}(1,:));
title(sprintf('person conf = %.3f', dets{15}(1,end)));
