function showboxesc(im, boxes, color, style)
% showboxes(im, boxes)
% Draw boxes on top of image.

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2009-2012 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

default_color = false;
if ~exist('color', 'var')
  default_color = true;
end

if isempty(im)
  hold on;
else
  image(im); 
  axis image;
  axis off;
end
if ~isempty(boxes)
  for j = 1:size(boxes,1)
    numfilters = floor(size(boxes, 2)/4);
    for i = 1:numfilters
      x1 = boxes(j,1+(i-1)*4);
      y1 = boxes(j,2+(i-1)*4);
      x2 = boxes(j,3+(i-1)*4);
      y2 = boxes(j,4+(i-1)*4);
      % remove unused filters
      del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
      x1(del) = [];
      x2(del) = [];
      y1(del) = [];
      y2(del) = [];

      if default_color 
        % 0 => diff
        % 1 => fn
        % 2 => tp
        style = '-';
        if boxes(j,end) == 0
          color = 'c';
        elseif boxes(j,end) == 1
          color = 'r';
        elseif boxes(j,end) == 2
          color = 'g';
        elseif boxes(j,end) == 3
          color = 'b';
        elseif boxes(j,end) == 4
          color = 'm';
          style = '--';
        end
      end

      line([x1 x1 x2 x2 x1 x1]', [y1 y2 y2 y1 y1 y2]', 'color', color, ...
                                                       'linewidth', 1, ...
                                                       'linestyle', style);
    end
  end
end
drawnow;
if isempty(im)
  hold off;
end

