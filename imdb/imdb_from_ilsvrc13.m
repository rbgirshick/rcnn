function imdb = imdb_from_ilsvrc13(root_dir, image_set)

% root_dir = '/work4/rbg/ILSVRC13';

% for each of the 200 classes there's a
%  train_pos_X
%  train_neg_X
%
% and there's also
%  val
%  test

% names
% ilsvrc13_val
% ilsvrc13_test
% ilsvrc13_train_pos_1
% ...
% ...
% ilsvrc13_train_pos_200
%
% split val into two folds with roughly equal # instances per class


%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/ILSVRC/ILSVRC2013_DET_train/n02672831/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'n02672831_11478', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'accordian', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

cache_file = ['./imdb/cache/imdb_ilsvrc13_' image_set];
try
  load(cache_file);
catch
  NUM_CLS = 200;
  bbox_path.train = fullfile(root_dir, 'ILSVRC2013_DET_bbox_train');
  bbox_path.val   = fullfile(root_dir, 'ILSVRC2013_DET_bbox_val');
  im_path.test    = fullfile(root_dir, 'ILSVRC2013_DET_test');
  im_path.train   = fullfile(root_dir, 'ILSVRC2013_DET_train');
  im_path.val     = fullfile(root_dir, 'ILSVRC2013_DET_val');
  im_path.val1    = fullfile(root_dir, 'ILSVRC2013_DET_val');
  im_path.val2    = fullfile(root_dir, 'ILSVRC2013_DET_val');
  devkit_path     = fullfile(root_dir, 'ILSVRC2013_devkit');
  meta_det        = load(fullfile(devkit_path, 'data', 'meta_det.mat'));

  imdb.name = ['ilsvrc13_' image_set];
  imdb.extension = 'JPEG';
  is_blacklisted = containers.Map;

  % derive image directory
  match = regexp(image_set, 'train_pos_(?<class_num>\d+)', 'names');
  if ~isempty(match)
    class_num = str2num(match.class_num);
    assert(class_num >= 1 && class_num <= NUM_CLS);
    imdb.image_dir = im_path.train;
    imdb.details.image_list_file = ...
        fullfile(devkit_path, 'data', 'det_lists', [image_set '.txt']);
    imdb.image_ids = textread(imdb.details.image_list_file, '%s');

    % only one class is present
    imdb.classes = {meta_det.synsets(class_num).name};
    imdb.num_classes = length(imdb.classes);
    imdb.class_to_id = ...
        containers.Map(imdb.classes, class_num);
    imdb.class_ids = class_num;

    imdb.image_at = @(i) ...
        fullfile(imdb.image_dir, get_wnid(imdb.image_ids{i}), ...
            [imdb.image_ids{i} '.' imdb.extension]);

    imdb.details.blacklist_file = [];
    imdb.details.bbox_path = bbox_path.train;
  elseif strcmp(image_set, 'val') || ...
      strcmp(image_set, 'val1') || ...
      strcmp(image_set, 'val2') || ...
      strcmp(image_set, 'test') 
    imdb.image_dir = im_path.(image_set);
    imdb.details.image_list_file = ...
        fullfile(devkit_path, 'data', 'det_lists', [image_set '.txt']);
    [imdb.image_ids, ~] = textread(imdb.details.image_list_file, '%s %d');

    % all classes are present
    imdb.classes = {meta_det.synsets(1:NUM_CLS).name};
    imdb.num_classes = length(imdb.classes);
    imdb.class_to_id = ...
        containers.Map(imdb.classes, 1:imdb.num_classes);
    imdb.class_ids = 1:imdb.num_classes;

    imdb.image_at = @(i) ...
        fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);

    if strcmp(image_set, 'val')
      imdb.details.blacklist_file = ...
          fullfile(devkit_path, 'data', ...
          'ILSVRC2013_det_validation_blacklist.txt');
      [bl_image_ids, bl_wnids] = textread(imdb.details.blacklist_file, '%d %s');
      is_blacklisted = containers.Map(bl_image_ids, ones(length(bl_image_ids), 1));
    else
      imdb.details.blacklist_file = [];
    end

    if ~strcmp(image_set, 'test')
      imdb.details.bbox_path = bbox_path.val;
    end
  else
    error('unknown image set');
  end

  % private ILSVRC 2013 details
  imdb.details.meta_det    = meta_det;
  imdb.details.root_dir    = root_dir;
  imdb.details.devkit_path = devkit_path;

  % VOC specific functions for evaluation and region of interest DB
  imdb.eval_func = @imdb_eval_ilsvrc13;
  imdb.roidb_func = @roidb_from_ilsvrc13;

  % Some images are blacklisted due to noisy annotations
  imdb.is_blacklisted = false(length(imdb.image_ids), 1);

  for i = 1:length(imdb.image_ids)
    tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
    try
      im = imread(imdb.image_at(i));
    catch
      lerr = lasterror;
      % gah, annoying data issues
      if strcmp(lerr.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
        warning('converting %s from CMYK to RGB', imdb.image_at(i));
        cmd = ['convert ' imdb.image_at(i) ' -colorspace CMYK -colorspace RGB ' ...
                imdb.image_at(i)];
        system(cmd);
        im = imread(imdb.image_at(i));
      else
        error(lerr.message);
      end
    end
    imdb.sizes(i, :) = [size(im, 1) size(im, 2)];
    imdb.is_blacklisted(i) = is_blacklisted.isKey(i);

    % faster, but seems to fail on some images :(
    %info = imfinfo(imdb.image_at(i));
    %assert(isscalar(info.Height) && info.Height > 0);
    %assert(isscalar(info.Width) && info.Width > 0);
    %imdb.sizes(i, :) = [info.Height info.Width];
  end

  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
