function VOCopts = get_voc_opts(path)

tmp = pwd;
voc_code_path = fullfile(path, 'VOCcode');
cd(voc_code_path);
try
  VOCinit; % brings VOCopts into scope
catch
  cd(tmp);
  error(sprintf('VOCcode directory not found at %s', voc_code_path));
end
cd(tmp);
