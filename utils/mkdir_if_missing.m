function made = mkdir_if_missing(path)
made = false;
if exist(path) == 0
  unix(['mkdir -p ' path]);
  made = true;
end
