function inds = subsample_images(imdb, num, seed)
if ~exist('seed', 'var')
  seed = [];
end
inds = find(imdb.is_blacklisted == false);
num = min(length(inds), num);
% fix the random seed for repeatability
prev_rng = seed_rand(seed);
inds = inds(randperm(length(inds), num));
% restore previous rng
rng(prev_rng);
