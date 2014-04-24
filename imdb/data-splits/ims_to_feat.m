function [X, D] = ims_to_feat(imdb, roidb)

X = zeros(length(imdb.image_ids), imdb.num_classes);
for i = 1:length(imdb.image_ids)
  if imdb.is_blacklisted(i)
    continue;
  end

  I_gt = find(roidb.rois(i).gt == true);
  gt_classes = roidb.rois(i).class(I_gt);

  counts = histc(gt_classes, 1:imdb.num_classes);

  X(i, 1:imdb.num_classes) = counts;
end

D = squareform(pdist(X, 'cityblock'));
D = D + (max(D(:))+1)*eye(size(D));

bl_inds = find(imdb.is_blacklisted);
D(:, bl_inds) = inf;
