function inds = subsample_images(imdb, num, seed)
if ~exist('seed', 'var')
  seed = [];
end
% fix the random seed for repeatability
prev_rng = seed_rand(seed);
num = min(length(imdb.image_ids), num);
inds = randperm(length(imdb.image_ids), num);
% restore previous rng
rng(prev_rng);
