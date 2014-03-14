function link_train_val(varargin)

ip = inputParser;
ip.addParamValue('year', @isstr);
ip.addParamValue('basedir', @isstr);

ip.parse(varargin{:});
opts = ip.Results;

conf = voc_config('pascal.year', opts.year);
VOCopts = conf.pascal.VOCopts;


start_dir = pwd;
cd(opts.basedir);
fprintf('switched to %s\n', pwd);

sets = {'train', 'val'};

for i = 1:length(sets)
  system(['mkdir ' sets{i}]);
  cd(sets{i});
  fprintf('switched to %s\n', pwd);
  image_ids = textread(sprintf(VOCopts.imgsetpath, sets{i}), '%s');
  for j = 1:length(image_ids)
    cmd = sprintf('ln -s ../trainval/%s.mat %s.mat', image_ids{j}, image_ids{j});
    fprintf([cmd '\n']);
    system(cmd);
  end
  cd('..');
  fprintf('switched to %s\n', pwd);
end

cd(start_dir);
fprintf('switched to %s\n', pwd);
