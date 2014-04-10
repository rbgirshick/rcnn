function res = imdb_eval_ilsvrc13(ignore, all_boxes, imdb, suffix)
% res = imdb_eval_ilsvrc13(ignore, all_boxes, imdb, suffix)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Delete results files after computing APs
rm_res = ~true;

% save results
if ~exist('suffix', 'var') || isempty(suffix) || strcmp(suffix, '')
  suffix = '';
else
  if suffix(1) ~= '_'
    suffix = ['_' suffix];
  end
end

conf = rcnn_config('sub_dir', imdb.name);

addpath(fullfile(imdb.details.devkit_path, 'evaluation')); 

pred_file = tempname();

% write out detections in ILSVRC format
fid = fopen(pred_file, 'w');
for cls = 1:length(all_boxes)
  tic_toc_print('writing out detections for class %d/%d\n', ...
      cls, length(all_boxes));
  boxes = all_boxes{cls};
  for image_index = 1:length(boxes);
    bbox = boxes{image_index};
    keep = nms(bbox, 0.3);
    bbox = bbox(keep,:);
    for j = 1:size(bbox,1)
      fprintf(fid, '%d %d %f %d %d %d %d\n', ...
          image_index, cls, bbox(j,end), bbox(j,1:4));
    end
  end
end
fclose(fid);

meta_file = fullfile(imdb.details.devkit_path, 'data', 'meta_det.mat');
eval_file = imdb.details.image_list_file;
blacklist_file = imdb.details.blacklist_file;

optional_cache_file = fullfile(imdb.details.root_dir, 'eval_det_cache', ...
    [imdb.name '.mat']);
mkdir_if_missing(fileparts(optional_cache_file));
gtruth_directory = imdb.details.bbox_path;

fprintf('pred_file: %s\n', pred_file);
fprintf('meta_file: %s\n', meta_file);

[ap, recall, precision] = eval_detection(pred_file, gtruth_directory, ...
    meta_file, eval_file, blacklist_file, optional_cache_file);

load(meta_file);
fprintf('-------------\n');
fprintf('Category\tAP\n');
for i = 1:200
    s = synsets(i).name;
    if length(s) < 8
        fprintf('%s\t\t%0.3f\n',s,ap(i));
    else
        fprintf('%s\t%0.3f\n',s,ap(i));
    end
end
fprintf(' - - - - - - - - \n');
fprintf('Mean AP:\t %0.3f\n',mean(ap));
fprintf('Median AP:\t %0.3f\n',median(ap));

res.recall = recall;
res.prec = precision;
res.ap = ap;

save([conf.cache_dir 'eval_detection_' imdb.name suffix], ...
    'res', 'recall', 'precision', 'ap');

if rm_res
  delete(pred_file);
end

rmpath(fullfile(imdb.details.devkit_path, 'evaluation')); 
