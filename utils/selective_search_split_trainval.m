function selective_search_split_trainval(year)

conf = voc_config('pascal.year', year);
VOCopts = conf.pascal.VOCopts;

train_image_ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');
val_image_ids = textread(sprintf(VOCopts.imgsetpath, 'val'), '%s');
trainval_image_ids = textread(sprintf(VOCopts.imgsetpath, 'trainval'), '%s');

trainval_image_id_map = containers.Map(trainval_image_ids, ...
    1:length(trainval_image_ids));

sels = load(['cachedir/convnet-selective-search/selective-search-data/voc_' ...
    year '_trainval']);

images = {};
boxes = {};
train_image_inds = trainval_image_id_map.values(train_image_ids);
for i = 1:length(train_image_ids)
  tic_toc_print('train: %d/%d\n', i, length(train_image_ids));
  ind = train_image_inds{i};
  images{i} = sels.images{ind};
  boxes{i} = sels.boxes{ind};
end

save(['cachedir/convnet-selective-search/selective-search-data/voc_' ...
    year '_train'], 'images', 'boxes');

images = {};
boxes = {};
val_image_inds = trainval_image_id_map.values(val_image_ids);
for i = 1:length(val_image_ids)
  tic_toc_print('val: %d/%d\n', i, length(val_image_ids));
  ind = val_image_inds{i};
  images{i} = sels.images{ind};
  boxes{i} = sels.boxes{ind};
end

save(['cachedir/convnet-selective-search/selective-search-data/voc_' ...
    year '_val'], 'images', 'boxes');
