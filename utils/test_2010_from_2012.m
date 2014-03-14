function test_2010_from_2012()

year = '2010';
testset = 'test';

VOCdevkit2012 = './datasets/VOCdevkit2012';
VOCdevkit2010 = './datasets/VOCdevkit2010';

imdb_2012 = imdb_from_voc(VOCdevkit2012, 'test', '2012');

image_ids_2010 = get_2010_test_image_ids();
detrespath_2010 = '/work4/rbg/VOC2010/VOCdevkit/results/VOC2010/Main/%s_det_test_%s.txt';
detrespath_2012 = imdb_2012.details.VOCopts.detrespath;

map = containers.Map;
for i = 1:length(image_ids_2010)
  map(image_ids_2010{i}) = true;
end

for i = 1:length(imdb_2012.details.VOCopts.classes)
  cls = imdb_2012.details.VOCopts.classes{i};
  res_fn = sprintf(detrespath_2012, 'comp4', cls);

  [ids, scores, x1, y1, x2, y2] = textread(res_fn, '%s %f %f %f %f %f');

  res_fn = sprintf(detrespath_2010, 'comp4', cls);

  % write out detections in PASCAL format and score
  fid = fopen(res_fn, 'w');
  for i = 1:length(ids)
    if map.isKey(ids{i})
      fprintf(fid, '%s %f %d %d %d %d\n', ids{i}, scores(i), x1(i), y1(i), x2(i), y2(i));
    end
  end
  fclose(fid);
end

function ids = get_2010_test_image_ids()
fn = '/work4/rbg/VOC2012/VOCdevkit/VOC2010/ImageSets/Main/test.txt';
ids = textread(fn, '%s');
