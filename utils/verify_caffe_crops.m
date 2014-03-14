function verify_caffe_crops()

CONTEXT_PAD = 16;
CROP_SIZE = 227;
opts.data_mean = '~/working/caffe-rbg/matlab/caffe/ilsvrc_2012_mean.mat';
d = load(opts.data_mean);
off = floor((256 - 227)/2)+1;
IMAGE_MEAN = d.image_mean(off:off+227-1, off:off+227-1, :);
clear d;
%IMAGE_MEAN(:) = 0;

s = dir('dump/*.txt');
for i = 1:2:length(s)

  info_file = s(i+1).name;
  data_file = s(i).name;
  info = textread(['dump/' info_file], '%s');
  im_path = info{1};
  x1 = str2num(info{2});
  y1 = str2num(info{3});
  x2 = str2num(info{4});
  y2 = str2num(info{5});
  flip = str2num(info{6});
  label = str2num(info{7});
  is_fg = str2num(info{8});
  ov = 0; %str2num(info{9});

  %M1 = textread(['dump/' data_file], '%f');
  fid = fopen(['dump/' data_file], 'rb');
  M1 = fread(fid, 227*227*3, 'single');
  fclose(fid);

  if length(M1) ~= 227*227*3
    warning('skipping wrong size image');
    continue;
  end

  im1 = reshape(M1, [227 227 3]);
  im1 = permute(im1, [2 1 3]);
  im1 = im1(:,:,[3 2 1]);
  im1 = (im1 - min(im1(:))) / (max(im1(:)) - min(im1(:)));
  subplot(1,2,1);
  imagesc(im1);
  axis image;
  title(sprintf('max %.2f min %.2f  label %d  ov %.3f', max(M1), min(M1), label, ov));

  im = imread(im_path);
  im = single(im(:,:,[3 2 1]));
  if CONTEXT_PAD > 0
    scale = CROP_SIZE/(CROP_SIZE - CONTEXT_PAD*2);
    bbox = [x1 y1 x2 y2];
    %figure(1); showboxesc(im/256, bbox, 'b', '-');
    height = bbox(4)-bbox(2)+1;
    width = bbox(3)-bbox(1)+1;
    center = [bbox(1)+width/2 bbox(2)+height/2];
    bbox = round([center center] + [-width -height width height]/2*scale);
    height = bbox(4)-bbox(2)+1;
    width = bbox(3)-bbox(1)+1;
    %figure(1); showboxesc([], bbox, 'r', '-');
    pad_x1 = max(0, 1 - bbox(1));
    pad_y1 = max(0, 1 - bbox(2));
    pad_x2 = max(0, bbox(3) - size(im,2));
    pad_y2 = max(0, bbox(4) - size(im,1));
    bbox(1) = max(1, bbox(1));
    bbox(2) = max(1, bbox(2));
    bbox(3) = min(size(im,2), bbox(3));
    bbox(4) = min(size(im,1), bbox(4));
    window = im(bbox(2):bbox(4), bbox(1):bbox(3), :);

    window_width = round((bbox(3)-bbox(1)+1)*CROP_SIZE/width);
    window_height = round((bbox(4)-bbox(2)+1)*CROP_SIZE/height);
    pad_x1 = round(pad_x1*CROP_SIZE/width);
    pad_y1 = round(pad_y1*CROP_SIZE/height);
    pad_x2 = round(pad_x2*CROP_SIZE/width);
    pad_y2 = round(pad_y2*CROP_SIZE/height);

    pad_h = pad_y1;
    pad_w = pad_x1;
    if flip
      pad_w = pad_x2;
    end

    if pad_h + window_height > CROP_SIZE
      window_height = CROP_SIZE - pad_h;
    end
    if pad_w + window_width > CROP_SIZE
      window_width = CROP_SIZE - pad_w;
    end
    tmp = imresize(window, [window_height window_width], 'bilinear', 'antialiasing', false);
    if flip
      tmp = tmp(:, end:-1:1, :);
    end
    tmp = tmp - IMAGE_MEAN(1+pad_h:window_height+pad_h, 1+pad_w:window_width+pad_w, :);
    %figure(2); window_ = tmp; imagesc((window_-min(window_(:)))/(max(window_(:))-min(window_(:)))); axis image;
    window = zeros(CROP_SIZE, CROP_SIZE, 3, 'single');
    window(1+pad_h:window_height+pad_h, 1+pad_w:window_width+pad_w, :) = tmp;
    %figure(3); imagesc((window-min(window(:)))/(max(window(:))-min(window(:)))); axis image; pause;
    window = permute(window, [2 1 3]);
  else
    window = im(y1:y2, x1:x2, :);
    fprintf('im: %s\n', im_path);
    fprintf('box: %d %d %d %d\n', x1, y1, x2, y2);
    window = imresize(window, [CROP_SIZE CROP_SIZE], 'bilinear', 'Antialiasing', false);
    if flip
      window = window(:, end:-1:1, :);
    end
    window = window - IMAGE_MEAN;
    % permute to make width the fastest dimension (for caffe)
    window = permute(window, [2 1 3]);
  end

  M2 = window(:);

  im2 = reshape(M2, [227 227 3]);
  im2 = permute(im2, [2 1 3]);
  im2 = im2(:,:,[3 2 1]);
  im2 = (im2 - min(im2(:))) / (max(im2(:)) - min(im2(:)));
  subplot(1,2,2);
  imagesc(im2);
  axis image;
  title(sprintf('max %.3f min %.3f', max(M2), min(M2)));

  fprintf('max diff: %f\n', max(abs(M1 - M2)));

  pause;
end
