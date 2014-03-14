function vis_crops(imdb)

opts.net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
opts.net_def_file = './model-defs/rcnn_batch_256_output_pool5.prototxt';

% load the region of interest database
roidb = imdb.roidb_func(imdb);

rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file);
rcnn_model = rcnn_load_model(rcnn_model);
image_mean = rcnn_model.cnn.image_mean;

im_perm = randperm(length(imdb.image_ids));

for i = im_perm
  d = roidb.rois(i);
  im = single(imread(imdb.image_at(i)));

  num_boxes = size(d.boxes, 1);

  crop_size = size(image_mean,1);

  perm = randperm(size(d.boxes, 1), 10);

  for j = perm
    bbox = d.boxes(j,:);
    src = im(bbox(2):bbox(4), bbox(1):bbox(3), :);
    crop_warp_0 = rcnn_im_crop(im, bbox, 'warp', crop_size, 0, image_mean);
    crop_warp_16 = rcnn_im_crop(im, bbox, 'warp', crop_size, 16, image_mean);
    crop_square_0 = rcnn_im_crop(im, bbox, 'square', crop_size, 0, image_mean);
    crop_square_16 = rcnn_im_crop(im, bbox, 'square', crop_size, 16, image_mean);

    max_val = max(cat(1, crop_warp_0(:), crop_warp_16(:), ...
        crop_square_0(:), crop_square_16(:)));
    min_val = min(cat(1, crop_warp_0(:), crop_warp_16(:), ...
        crop_square_0(:), crop_square_16(:)));

    src = normalize(src, max(src(:)), min(src(:)));
    crop_warp_0 = normalize(crop_warp_0, max_val, min_val);
    crop_warp_16 = normalize(crop_warp_16, max_val, min_val);
    crop_square_0 = normalize(crop_square_0, max_val, min_val);
    crop_square_16 = normalize(crop_square_16, max_val, min_val);

    subplot(2, 4, 1);
    imagesc(src);
    title('src');
    axis image;
    axis off;

    subplot(2, 4, 5);
    imagesc(crop_warp_0);
    title('warp 0');
    axis image;
    axis off;

    subplot(2, 4, 6);
    imagesc(crop_warp_16);
    title('warp 16');
    axis image;
    axis off;

    subplot(2, 4, 7);
    imagesc(crop_square_0);
    title('square 0');
    axis image;
    axis off;

    subplot(2, 4, 8);
    imagesc(crop_square_16);
    title('square 16');
    axis image;
    axis off;

    pause;
  end
end


function A = normalize(A, max_val, min_val)
range = max_val - min_val;
A = (A - min_val) / (range + eps);
