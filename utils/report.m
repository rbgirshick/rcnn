function report(dir1, suffix1, showcls, do_auc_ap)
% Print scores for all classes.

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

if nargin < 3
  showcls = true;
end

if nargin < 4
  do_auc_ap = false;
end

files = dir([dir1 '/*' suffix1 '.mat']);
classes = cell(length(files), 1);
for i = 1:length(files)
  j = strfind(files(i).name, '_');
  classes{i} = files(i).name(1:j(1)-1);
end

count1 = 0;
for i=1:length(classes)
  cls = classes{i};
  try
    load([dir1 cls suffix1]);
    ap1 = ap;
    if do_auc_ap 
      ap2 = xVOCap(recall, prec);
      score2(i) = ap2;
    end
    if showcls
      if do_auc_ap
        fprintf('%12s %.4f\n', cls, ap2);
      else
        fprintf('%12s %.4f\n', cls, ap1);
      end
    else
      if do_auc_ap
        fprintf('%.4f\n', ap2);
      else
        fprintf('%.4f\n', ap1);
      end
    end
    score1(i) = ap1;
  catch
    score1(i) = nan;
    score2(i) = nan;
    if showcls
      fprintf('%12s -\n', cls);
    else
      fprintf('-\n');
    end
  end
end

a1 = nanmean(score1);
if do_auc_ap
  a2 = nanmean(score2);
end
if showcls
  fprintf('%s\n', repmat('-', [1 12]));
  if do_auc_ap
    fprintf('%12s %.4f\n', 'mAP', a2);
  else
    fprintf('%12s %.4f\n', 'mAP', a1);
  end
else
  if do_auc_ap
    fprintf('%.4f\n', a2);
  else
    fprintf('%.4f\n', a1);
  end
end
