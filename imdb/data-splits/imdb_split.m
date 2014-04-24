function imdb_split(n, imdb, roidb)

if ~exist('roidb', 'var') || isempty(roidb)
  roidb = imdb.roidb_func(imdb);
end

save_file = '/var/tmp/foo.mat';
try
  load(save_file);
catch
  [X, D] = ims_to_feat(imdb, roidb);
  save(save_file, 'X', 'D', '-v7.3');
end

trials = 1;
good = 1;
best_b = inf;
while good <= n
  fprintf('trial %d (%.3f)\n', trials, best_b)
  [t_ids, v_ids] = split(X, D, imdb);
  [t_counts, v_counts, totals] = class_stats(X, t_ids, v_ids);
  [b, t_counts, v_counts, t_ids, v_ids] = local_search(X, t_ids, v_ids, t_counts, v_counts, totals);
  ratios = t_counts ./ totals;
  if b < best_b
    fprintf('new min value: %.3f\n', b);
    best_b = b;
    write_ids(['imdb/data-splits/det_lists/splitA_' num2str(good) '.txt'], t_ids, imdb);
    write_ids(['imdb/data-splits/det_lists/splitB_' num2str(good) '.txt'], v_ids, imdb);

    fprintf('found a good one (up to %d)\n', good);
    %disp([t_counts' v_counts' totals' abs(t_counts-v_counts)' 100*abs(t_counts-v_counts)'./totals']);
    fprintf('obj: %.3f  min: %.3f  max: %.3f  mean: %.3f  median: %.3f\n', ...
        b, min(ratios), max(ratios), mean(ratios), median(ratios));

    good = good + 1;
  end
  trials = trials + 1;
end


% ------------------------------------------------------------------------
function [t_ids, v_ids] = split(X, D, imdb)
% ------------------------------------------------------------------------
t_ids = [];
v_ids = [];

num = size(D, 1);
picked = false(num, 1);
bl_inds = find(imdb.is_blacklisted);
% prevent picking blacklisted rows
picked(bl_inds) = true;

rs = RandStream('mt19937ar','Seed','shuffle');
perm = randperm(rs, num);

for i = perm
  if picked(i)
    continue;
  end
  picked(i) = true;

  [min_val, j] = min(D(i,:));
  assert(min_val ~= inf);
  D(:,i) = inf;
  D(:,j) = inf;
  picked(j) = true;

  if rand() < 0.5
    t_ids(end+1) = i;
    v_ids(end+1) = j;
  else
    t_ids(end+1) = j;
    v_ids(end+1) = i;
  end
end




function [bmax, t_counts, v_counts, t_ids, v_ids] = local_search(X, t_ids, v_ids, t_counts, v_counts, totals)

progress = true;
rs = RandStream('mt19937ar','Seed','shuffle');
while progress
  progress = false;

  perm = randperm(rs, length(t_ids));
  bmax = max(abs(t_counts - v_counts) ./ totals);
  bavg = mean(abs(t_counts - v_counts) ./ totals);

  mv = [];
  for i = perm
    j = t_ids(i);
    counts = X(j,:);
    tc = t_counts - counts;
    vc = v_counts + counts;

    new_bmax = max(abs(tc - vc) ./ totals);
    new_bavg = mean(abs(tc - vc) ./ totals);

    if new_bmax < bmax && new_bavg < bavg
      t_counts = tc;
      v_counts = vc;
      bmax = new_bmax;
      bavg = new_bavg;
      mv = [mv; i];
      progress = true;
      %fprintf('reduced bmax to %f\n', bmax);
    end
  end
  v_ids = cat(2, v_ids, t_ids(mv));
  t_ids(mv) = [];

  perm = randperm(rs, length(v_ids));
  bmax = max(abs(t_counts - v_counts) ./ totals);
  bavg = mean(abs(t_counts - v_counts) ./ totals);

  mv = [];
  for i = perm
    j = v_ids(i);
    counts = X(j,:);
    tc = t_counts + counts;
    vc = v_counts - counts;

    new_bmax = max(abs(tc - vc) ./ totals);
    new_bavg = mean(abs(tc - vc) ./ totals);

    if new_bmax < bmax && new_bavg < bavg
      t_counts = tc;
      v_counts = vc;
      bmax = new_bmax;
      bavg = new_bavg;
      mv = [mv; i];
      progress = true;
      %fprintf('reduced bmax to %f\n', bmax);
    end
  end
  t_ids = cat(2, t_ids, v_ids(mv));
  v_ids(mv) = [];
end




function write_ids(name, ids, imdb)

fid = fopen(name, 'w');
for i = 1:length(ids)
  fprintf(fid, '%s %d\n', imdb.image_ids{ids(i)}, i);
end
fclose(fid);


function [t_counts, v_counts, totals] = class_stats(class_counts, t_inds, v_inds)
t_counts = sum(class_counts(t_inds, :));
v_counts = sum(class_counts(v_inds, :));
totals = t_counts + v_counts;
