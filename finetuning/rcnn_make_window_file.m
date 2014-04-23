function rcnn_make_window_file(imdb, out_dir, window_file_name, num_to_sample)
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

if length(imdb) == 1 && ...
    (~exist('window_file_name', 'var') || isempty(window_file_name))
  window_file = sprintf('%s/window_file_%s.txt', ...
      out_dir, imdb.name);
else
  assert(exist('window_file_name', 'var') && ~isempty(window_file_name));
  window_file = sprintf('%s/window_file_%s.txt', ...
      out_dir, window_file_name);
end
fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('writing window file to: %s\n', window_file)
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');
fid = fopen(window_file, 'wt');

channels = 3; % three channel images

skip = 0;
image_index = 0;
for ii = 1:length(imdb)
  roidb = imdb(ii).roidb_func(imdb(ii));

  match = regexp(imdb(ii).name, 'train_pos_(?<class_num>\d+)', 'names');
  if ~isempty(match)
    class_num = str2num(match.class_num);
    inds_to_sample = subsample_images(imdb(ii), num_to_sample, class_num);
  else
    inds_to_sample = 1:length(imdb(ii).image_ids);
  end

  for iii = 1:length(inds_to_sample)
    i = inds_to_sample(iii);
    tic_toc_print('make window file (%s): %d/%d\n', ...
        imdb(ii).name, iii, length(inds_to_sample));
    img_path = imdb(ii).image_at(i);
    roi = roidb.rois(i);
    num_boxes = size(roi.boxes, 1);
    if num_boxes > 0
      fprintf(fid, '# %d\n', image_index);
      fprintf(fid, '%s\n', img_path);
      fprintf(fid, '%d\n%d\n%d\n', ...
          channels, ...
          imdb(ii).sizes(i, 1), ...
          imdb(ii).sizes(i, 2));
      fprintf(fid, '%d\n', num_boxes);

      [ovs, labels] = max(roi.overlap, [], 2);
      I = find(ovs < 1e-5);
      ovs(I) = 0;
      labels(I) = 0;
      bboxes = round(roi.boxes-1);
      bboxes(:,1) = max(0, bboxes(:,1));
      bboxes(:,2) = max(0, bboxes(:,2));
      bboxes(:,3) = min(imdb(ii).sizes(i, 2)-1, bboxes(:,3));
      bboxes(:,4) = min(imdb(ii).sizes(i, 1)-1, bboxes(:,4));

      data = [labels full(ovs) bboxes];
      fprintf(fid, '%d %.3f %d %d %d %d\n', data');
      image_index = image_index + 1;
    end
  end
end
fclose(fid);
