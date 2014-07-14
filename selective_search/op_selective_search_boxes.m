function result = op_selective_search_boxes(first_el, last_el, imdb)
% result = op_selective_search_boxes(first_el, last_el, imdb)
%
% Loop slice operation for distributed computation of selective search
% boxes.
%
% This function depends on simple-cluster-lib, which is specific to 
% the Berkeley cluster and not useful to the general public (and 
% hence not available). This file exists because it's convenient for 
% me to keep it in the repository.
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

fast_mode = true;
mean_num = 0;
mean_time = 0;

result = cell(last_el-first_el+1, 1);
for i = first_el:last_el
  i_ = i-first_el+1;

  fprintf('%d/%d (%s) ...', i, last_el, imdb.image_ids{i});
  im = imread(imdb.image_at(i));
  th = tic();
  result{i_} = selective_search_boxes(im, fast_mode);
  t = toc(th);

  mean_num = (mean_num * (i_-1) + size(result{i_}, 1))/i_;
  mean_time = (mean_time * (i_-1) + t)/i_;
  fprintf('%.2fs...%d boxes (means: %.2fs %.1f boxes)\n', t, ...
      size(result{i_}, 1), mean_time, mean_num);
end
