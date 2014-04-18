function rcnn_make_window_file(imdb, out_dir)
% rcnn_make_window_file(imdb, out_dir)
%   Makes a window file that can be used by the caffe WindowDataLayer 
%   for finetuning.
%
%   The window file format contains repeated blocks of:
%
%     # image_index 
%     img_path
%     channels 
%     height 
%     width
%     num_windows
%     class_index overlap x1 y1 x2 y2
%     <... num_windows-1 more windows follow ...>

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

roidb = imdb.roidb_func(imdb);

window_file = sprintf('%s/window_file_%s.txt', ...
    out_dir, imdb.name);
fid = fopen(window_file, 'wt');

channels = 3; % three channel images

skip = 0;
image_index = 0;
for i = 1:length(imdb.image_ids)
  tic_toc_print('make window file: %d/%d\n', i, length(imdb.image_ids));
  img_path = imdb.image_at(i);
  roi = roidb.rois(i);
  num_boxes = size(roi.boxes, 1);
  %im = imread(img_path);
  if 0 && size(im,3) ~= channels
    skip = skip + 1;
    fprintf('skipping ~3-chan image (%.3f)\n', skip/i*100);
    continue;
  end
  fprintf(fid, '# %d\n', image_index);
  fprintf(fid, '%s\n', img_path);
  fprintf(fid, '%d\n%d\n%d\n', ...
      channels, ...
      imdb.sizes(i, 1), ...
      imdb.sizes(i, 2));
  fprintf(fid, '%d\n', num_boxes);

  [ovs, labels] = max(roi.overlap, [], 2);
  I = find(ovs < 1e-5);
  ovs(I) = 0;
  labels(I) = 0;
  bboxes = round(roi.boxes-1);
  bboxes(:,1) = max(0, bboxes(:,1));
  bboxes(:,2) = max(0, bboxes(:,2));
  bboxes(:,3) = min(imdb.sizes(i, 2)-1, bboxes(:,3));
  bboxes(:,4) = min(imdb.sizes(i, 1)-1, bboxes(:,4));

  data = [labels full(ovs) bboxes];
  fprintf(fid, '%d %.3f %d %d %d %d\n', data');

%  % TODO: vectorize the following very slow loop
%  buffer = [];
%  for j = 1:num_boxes
%    [ov, label] = max(roi.overlap(j,:));
%    % zero overlap => label = 0 (background)
%    if ov < 1e-5
%      label = 0;
%      ov = 0;
%    end
%    % make sure boxes are 0-based integers within the image
%    bbox = round(roi.boxes(j,:)-1);
%    bbox(1) = max(0, bbox(1));
%    bbox(2) = max(0, bbox(2));
%    bbox(3) = min(imdb.sizes(i, 2)-1, bbox(3));
%    bbox(4) = min(imdb.sizes(i, 1)-1, bbox(4));
%    buffer = [buffer ...
%        sprintf('%d %.3f %d %d %d %d\n', ...
%            label, full(ov), bbox(1), bbox(2), bbox(3), bbox(4))];
%  end
%  fprintf(fid, buffer);

  image_index = image_index + 1;
end

fclose(fid);
