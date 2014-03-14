function [res, models] = grid_search(train_split, train_year, test_split, test_year, layer, fine_tuned)

% layer and fine_tuned are used to point to the correct feature cache
% modify train_classes
%  - load features from feature cache "raw"
%  - save them "raw"
%  - after loading saved features, apply current pwr_xform

pwr_xforms       = 1;
svm_Cs           = [10^-3 10^-3.5 10^-2.5]; %10.^(-2.5:-0.5:-3.5);
%svm_Cs           = [10^-3.5 10^-4]; %10.^(-2.5:-0.5:-3.5);
%svm_Cs           = [10^-2]; %10.^(-2.5:-0.5:-3.5);
pos_loss_weights = [2 1];

% common config
opts.bias_mult   = 10;
opts.layer       = layer;
opts.fine_tuned  = fine_tuned;
opts.use_flipped = 0;
subdir = sprintf('%s_%s_vs_%s_%s_layer_%s_finetuned_%d_caffe', ...
                 train_split, train_year, test_split, ...
                 test_year, layer, fine_tuned);

res = {};
models = {};

global VOC_CONFIG_OVERRIDE;

for pwr_xform = pwr_xforms
  for pos_loss_weight = pos_loss_weights
    for svm_C = svm_Cs
      fprintf('%s: pwr: %2.2f  C: %5.5f  w1: %d\n', subdir, pwr_xform, svm_C, pos_loss_weight);

      % set an override to make a project directory specific to the outputs of this
      conf_override.project = sprintf('%s/%s/pos_gt_neg_0.3_pwr_%2.2f_svm_C_%5.5f_w1_%d', ...
                                      'convnet-selective-search/grid-search', ...
                                      subdir, pwr_xform, svm_C, pos_loss_weight);
      VOC_CONFIG_OVERRIDE = @() conf_override;

      % opts for this grid search point
      opts.pwr_xform       = pwr_xform;
      opts.svm_C           = svm_C;
      opts.pos_loss_weight = pos_loss_weight;
      
      % train and test
      conf = voc_config();

      save_file = [conf.paths.model_dir 'models_final'];
      try
        ld = load(save_file);
        models{end+1} = ld.models;
        clear ld;
      catch
        models{end+1} = train_classes(conf.pascal.VOCopts.classes, train_split, train_year, ...
                                      'bias_mult', opts.bias_mult, ...
                                      'layer', opts.layer, ...
                                      'fine_tuned', opts.fine_tuned, ...
                                      'use_flipped', opts.use_flipped, ...
                                      'svm_C', opts.svm_C, ...
                                      'pos_loss_weight', opts.pos_loss_weight);
        % saved in train_classes
      end

      try
        for i = 1:length(models)
          fn = [conf.paths.model_dir models{end}{i}.class '_boxes_' test_split '_' test_year '.mat'];
          if ~exist(fn)
            fprintf('%s does not exist\n', fn);
            error();
          end
        end
      catch
        res{end+1} = test_classes(models{end}, test_split, test_year);
      end
    end
  end
end
