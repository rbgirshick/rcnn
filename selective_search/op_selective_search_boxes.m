function result = op_selective_search_boxes(first_el, last_el, imdb)

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
