function roidb_browser(imdb, class_id, roidb)

if ~exist('roidb', 'var') || isempty(roidb)
  roidb = imdb.roidb_func(imdb);
end

for i = 1:length(roidb.rois)
  I_gt = find(roidb.rois(i).gt == true & roidb.rois(i).class == class_id);
  if ~isempty(I_gt)
    im = imread(imdb.image_at(i));
    if size(im, 3) == 1
      im = repmat(im, [1 1 3]);
    end
    showboxes(im, roidb.rois(i).boxes(I_gt, :));
    fprintf('\n%d/%d: %s\n', i, length(roidb.rois), imdb.image_ids{i});
    if imdb.is_blacklisted(i)
      title('BLACKLISTED');
    end
    pause;
  else
    fprintf('.');
  end
end
