function d = rcnn_load_cached_pool5_features(cache_name, imdb_name, id)
% loads feat_cache/[cache_name]/[split]/[id].mat

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
