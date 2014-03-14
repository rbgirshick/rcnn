function [best_thresh, best_ap, res] = nms_tune_threshold(cls, testset, year)

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
cachedir = conf.paths.model_dir;                  
VOCopts  = conf.pascal.VOCopts;
load([cachedir cls '_boxes_' testset '_' year]);

threshs = 0.1:0.05:0.5;
best_ap = 0;

res = zeros(length(threshs), 2);
for i = 1:length(threshs)
  thresh = threshs(i);
  ap_auc = compute_at_nms_thresh(cls, boxes, thresh, VOCopts);
  res(i,:) = [thresh ap_auc];
  if ap_auc > best_ap
    best_thresh = thresh;
    best_ap = ap_auc;
    fprintf('!!! %s AP = %.3f @ thresh = %.3f\n', cls, best_ap, thresh);
  end
end



function ap_auc = compute_at_nms_thresh(cls, boxes, thresh, VOCopts)

image_ids = textread(sprintf(VOCopts.imgsetpath, VOCopts.testset), '%s');
% write out detections in PASCAL format and score
fid = fopen(sprintf(VOCopts.detrespath, 'comp3', cls), 'w');
for i = 1:length(image_ids);
  bbox = boxes{i};
  keep = nms(bbox, thresh);
  bbox = bbox(keep,:);
  for j = 1:size(bbox,1)
    fprintf(fid, '%s %f %d %d %d %d\n', image_ids{i}, bbox(j,end), bbox(j,1:4));
  end
end
fclose(fid);

tic;
[recall, prec, ap_auc] = x10VOCevaldet(VOCopts, 'comp3', cls, true);
%ap_auc = xVOCap(recall, prec);
