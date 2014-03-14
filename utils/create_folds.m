function folds = create_folds(N, k_folds)

prev_rng = seed_rand();

perm = randperm(N);
fold_points = floor(linspace(1, N, k_folds+1));
fold_points(end) = N+1;
folds = cell(k_folds, 1);
for i = 1:k_folds
  folds{i} = perm(fold_points(i):fold_points(i+1)-1);
end
assert(isempty(setdiff(1:N, cat(2, folds{:}))));

rng(prev_rng);
