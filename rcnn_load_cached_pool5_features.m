function d = rcnn_load_cached_pool5_features(cache_name, imdb_name, id)
% d = rcnn_load_cached_pool5_features(cache_name, imdb_name, id)
%   loads cached pool5 features from:
%   feat_cache/[cache_name]/[imdb_name]/[id].mat

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------


file = sprintf('./feat_cache/%s/%s/%s', cache_name, imdb_name, id);

if exist([file '.mat'], 'file')
  d = load(file);
else
  warning('could not load: %s', file);
  d = create_empty();
end

% standardize boxes to double (for overlap calculations, etc.)
d.boxes = double(d.boxes);


% ------------------------------------------------------------------------
function d = create_empty()
% ------------------------------------------------------------------------
d.gt = logical([]);
d.overlap = single([]);
d.boxes = single([]);
d.feat = single([]);
d.class = uint8([]);
