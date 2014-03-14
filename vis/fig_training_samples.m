function imgs = fig_trainig_samples(split, year)

% for each class, pick 10 examples at random

conf = voc_config('pascal.year', year);

% Set up training dataset
image_ids = textread(sprintf(conf.pascal.VOCopts.imgsetpath, split), '%s');

perm = randperm(length(image_ids));

imgs = cell(21, 1);
for i = 1:21
  imgs{i} = zeros(224, 224, 3, 0);
end

opts.layer = 'fc6';
opts.fine_tuned = 0;
opts.use_flipped = 0;

done = false(20, 1);
for i = 1:length(perm)
  d = load_cached_features(split, year, image_ids{perm(i)}, opts);
  im = imread(sprintf(conf.pascal.VOCopts.imgpath, image_ids{perm(i)})); 

  lens = cellfun(@(x) size(x,4), imgs, 'UniformOutput', false);
  disp(lens);

  for j = 1:20
    I_gt = find(d.class == j);
    if ~isempty(I_gt)
      I_gt = I_gt(randperm(length(I_gt), 1));
      bbox = d.boxes(I_gt, :);
      imgs{j} = cat(4, imgs{j}, imresize(im(bbox(2):bbox(4), bbox(1):bbox(3), :), [224 224]));
    end
    if size(imgs{j}, 4) >= 10
      done(j) = true;
    end
  end

  if size(imgs{21}, 4) < 18
    I_bg = find(d.class == 0 & max(d.overlap,[],2) < 0.1);
    I_bg = I_bg(randperm(length(I_bg), 1));
    bbox = d.boxes(I_bg, :);
    imgs{21} = cat(4, imgs{21}, imresize(im(bbox(2):bbox(4), bbox(1):bbox(3), :), [224 224]));
  end

  if all(done)
    break;
  end
end

% pick 10 windows at random
for i = 1:20
  clf;
  montage(imgs{i}(:,:,:,1:9), 'Size', [3 3])
  set(gcf, 'Color', 'white')
  export_fig(sprintf('paper-figures/warped-samples/%d.pdf', i));
end

clf;
montage(imgs{21}(:,:,:,1:18), 'Size', [3 6])
set(gcf, 'Color', 'white')
export_fig(sprintf('paper-figures/warped-samples/%d.pdf', 21));

