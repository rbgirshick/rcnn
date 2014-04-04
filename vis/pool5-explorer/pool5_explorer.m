function pool5_explorer(imdb, cache_name)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

conf = rcnn_config('sub_dir', imdb.name);

index = pool5_explorer_build_index(imdb, cache_name);

figures = [1 2];
vf_sz = [8 12];
redraw = true;
position = 1;

channel = 0;
cell_y = 2;
cell_x = 2;
feature_index = get_feature_index(channel, cell_y, cell_x);

while 1
  if redraw
    feature_index = get_feature_index(channel, cell_y, cell_x);
    visualize_feature(imdb, figures(2), index, feature_index, position, ...
        vf_sz, cell_y, cell_x, channel);
    redraw = false;
  end

  % display 16x16 grid where each cell shows a 6x6 feature map

  % wait for mouse click
  %  -> map click coordinates to feature dimension

  [~, ~, ~, ~, key_code] = get_feature_selection(1);

  switch key_code
    case 27 % ESC
      close(figures(ishandle(figures)));
      return;

    case '`'
      return;

    case 's' % take a snapshot
      filename = sprintf('./vis/pool5-explorer/shots/x%d-y%d-c%d-p%d.pdf', ...
                         cell_y, cell_x, channel, position);
      if exist(filename)
        delete(filename);
      end
      export_fig(filename);

    case 'g' % go to a specific channel
      answer = str2double(inputdlg('go to channel:'));
      if ~isempty(answer)
        answer = round(answer);
        if answer > 0
          channel = answer - 1;
          redraw = true;
        end
      end

    case 31 % up
      % decrease channel
      if channel > 0
        channel = channel - 1;
        position = 1;
        redraw = true;
      end

    case 30 % down
      % increase channel
      if channel < 255
        channel = channel + 1;
        position = 1;
        redraw = true;
      end

    case 'i' % cell up
      if cell_y > 0
        cell_y = cell_y - 1;
        position = 1;
        redraw = true;
      end

    case 'k' % cell down
      if cell_y < 5
        cell_y = cell_y + 1;
        position = 1;
        redraw = true;
      end

    case 'j' % cell left
      if cell_x > 0
        cell_x = cell_x - 1;
        position = 1;
        redraw = true;
      end

    case 'l' % cell right
      if cell_x < 5
        cell_x = cell_x + 1;
        position = 1;
        redraw = true;
      end

    case 29 % ->
      new_pos = position + prod(vf_sz);
      if new_pos < length(index.features{feature_index}.scores)
        position = new_pos;
        redraw = true;
      end

    case 28 % <-
      new_pos = position - prod(vf_sz);
      if new_pos > 0
        position = new_pos;
        redraw = true;
      end
    otherwise
      fprintf('%d\n', key_code);
  end
end


% ------------------------------------------------------------------------
function f = get_feature_index(channel, cell_y, cell_x)
% ------------------------------------------------------------------------
f = channel*36 + cell_y*6 + cell_x + 1;


% ------------------------------------------------------------------------
function visualize_feature(imdb, fig, index, f, position, msz, cell_y, cell_x, channel)
% ------------------------------------------------------------------------
max_val = 0;
for x_ = 0:5
  for y_ = 0:5
    f_ = get_feature_index(channel, y_, x_);
    max_val = max([max_val; index.features{f_}.scores]);
  end
end

s = 227/6;
points = round(s/2:s:227);

M = zeros(6,6,256);
M(f) = 1;
M = sum(M, 3)';

half_receptive_field = floor(195/2);
[r,c] = find(M);
r1 = max(1, points(r) - half_receptive_field);
r2 = min(227, points(r) + half_receptive_field);
c1 = max(1, points(c) - half_receptive_field);
c2 = min(227, points(c) + half_receptive_field);
h = r2-r1;
w = c2-c1;

psx = 96;
psy = 96;
h = h * psy/227;
w = w * psx/227;
context_padding = round(16/227 * 96);

r1 = (r1-1)*psy/227 + 1;
c1 = (c1-1)*psx/227 + 1;

ims = {};
start_pos = position;
end_pos = min(length(index.features{f}.scores), start_pos + prod(msz) - 1);
N = end_pos - start_pos + 1;
str = sprintf('pool5 feature: (%d,%d,%d) (top %d - %d)', cell_y+1, cell_x+1, channel+1, start_pos, end_pos);
for i = start_pos:end_pos
  val = index.features{f}.scores(i);
  image_ind = index.features{f}.image_inds(i);
  bbox = index.features{f}.boxes(i, :);

  im = imread(imdb.image_at(image_ind));
  im = rcnn_im_crop(im, bbox, 'warp', psx, context_padding, []);

  ims{end+1} = uint8(im);
end
filler = prod(msz) - N;
im = my_montage(cat(4, ims{:}, 256*ones(psy, psx, 3, filler)), msz);
figure(2);
clf;
imagesc(im);
title(str, 'Color', 'black', 'FontSize', 18, 'FontName', 'Times New Roman');
axis image;
axis off;
set(gcf, 'Color', 'white');
q = 1;
for y = 0:msz(1)-1
  for x = 0:msz(2)-1
    if q > N
      break;
    end
    x1 = c1+psx*x;
    y1 = r1+psy*y;
    rectangle('Position', [x1 y1 w h], 'EdgeColor', 'w', 'LineWidth', 3);
    text(x1, y1+7.5, sprintf('%.1f', index.features{f}.scores(start_pos+q-1)/max_val), 'BackgroundColor', 'w', 'FontSize', 10, 'Margin', 0.1, 'FontName', 'Times New Roman');
    q = q + 1;
  end
  if q > N
    break;
  end
end

if 0
  % compute mean figure
  num_to_avg = 40;
  scores = index.features{f}.scores(start_pos:end_pos);
  for i = 1:num_to_avg
    ims{i} = double(ims{i})*scores(i)/sum(scores(1:num_to_avg));
  end
  figure(1);
  imagesc(uint8(sum(cat(4, ims{1:num_to_avg}), 4)));
  axis image;
  figure(2);
end


% ------------------------------------------------------------------------
function [feature_index, channel, cell_y, cell_x, ch] = ...
    get_feature_selection(channel_width)
% ------------------------------------------------------------------------
while 1
  [x,y,ch] = ginput(1);
  chan_y = floor(y/channel_width);
  chan_x = floor(x/channel_width);
  channel = chan_y*16 + chan_x;

  cell_y = floor(rem(y, channel_width)/7);
  cell_x = floor(rem(x, channel_width)/7);

  feature_index = channel*36 + cell_y*6 + cell_x + 1;

  if (channel < 0 || channel > 255)
    channel = nan;
  end
  if isscalar(ch)
    return;
  end
end


% ------------------------------------------------------------------------
function im = my_montage(ims, sz)
% ------------------------------------------------------------------------
ims_sz = [size(ims, 1) size(ims, 2)];
im = zeros(ims_sz(1)*sz(1), ims_sz(2)*sz(2), 3, class(ims));
k = 1;
for y = 0:sz(1)-1
  for x = 0:sz(2)-1
    im(y*ims_sz(1)+1:(y+1)*ims_sz(1), ...
       x*ims_sz(2)+1:(x+1)*ims_sz(2), :) = ims(:,:,:,k);
    k = k + 1;
  end
end
