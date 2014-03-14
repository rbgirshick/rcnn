function cmp_feat_caches(dir1, dir2)
% For comparing new feat cache files to known good
% feat cache files after some change

s = dir([dir1 '/*.mat']);

for i = 1:length(s)
  [a,b,c] = fileparts(s(i).name);
  s1 = [dir1 '/' s(i).name];
  s2 = [dir2 '/' b '.mat'];
  if ~exist(s2)
    fprintf('Skipping %s\n', s(i).name);
    continue;
  end

  d1 = load(s1);
  d2 = load(s2);

  %  gt: [2448x1 logical]
  %  overlap: [2448x20 single]
  %  boxes: [2448x4 single]
  %  feat: [2448x9216 single]
  %  class: [2448x1 uint8]

  assert(sum(abs(d1.gt - d2.gt)) == 0);
  assert(sum(sum(abs(d1.overlap - d2.overlap))) == 0);
  assert(sum(sum(abs(d1.boxes - d2.boxes))) == 0);
  assert(sum(sum(abs(d1.feat - d2.feat))) == 0);
  assert(sum(abs(d1.class - d2.class)) == 0);

  fprintf('%d is ok\n', i);
end
