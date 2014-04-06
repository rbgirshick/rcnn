function [recall, recalled, count] = roidb_compute_recall(roidb, thresh)

recalled = 0;
count = 0;
for i = 1:length(roidb.rois)
  tic_toc_print('compute recall: %d/%d\n', i, length(roidb.rois));
  if isfield(roidb.rois(i), 'is_difficult')
    I_gt = find(roidb.rois(i).gt == true & ~roidb.rois(i).is_difficult);
  else
    I_gt = find(roidb.rois(i).gt == true);
  end
  I_non_gt = find(roidb.rois(i).gt == false);
  gt_boxes = roidb.rois(i).boxes(I_gt, :);
  non_gt_boxes = roidb.rois(i).boxes(I_non_gt, :);
  for j = 1:length(I_gt)
    max_iou = max(boxoverlap(non_gt_boxes, gt_boxes(j, :)));
    if max_iou >= thresh
      recalled = recalled + 1;
    end
  end
  count = count + length(I_gt);
end

recall = recalled / count;
