function rcnn_run_pool5_explorer()

VOCdevkit = '/work4/rbg/VOC2007/VOCdevkit';

imdb = imdb_from_voc(VOCdevkit, 'test', '2007');
pool5_explorer(imdb, 'v1_finetune_voc_2007_trainval_iter_70000');
