## Contents -- Precomputed R-CNN Models

### R-CNN detectors trained on PASCAL VOC
* `./rcnn_models/voc_2007/rcnn_model_not_finetuned`
  * The CNN for this detector was *not* fine-tuned (it was trained only on ILSVRC 2012 train)
  * Uses: `./caffe_nets/ilsvrc_2012_train_iter_310k`
  * This detector was trained on PASCAL VOC 2007 trainval
* `./rcnn_models/voc_2007/rcnn_model_finetuned`
  * The CNN for this detector was fine-tuned on PASCAL VOC 2007 trainval
  * Uses: `./caffe_nets/finetune_voc_2007_trainval_iter_70k`
  * This detector was trained on PASCAL VOC 2007 trainval
* `./rcnn_models/voc_2012/rcnn_model_finetuned`
  * The CNN for this detector was fine-tuned on PASCAL VOC 2012 train
  * Uses: `./caffe_nets/voc_2012/finetune_voc_2012_train_iter_70k`
  * This detector was trained on PASCAL VOC 2012 **train** (val was not used)
* `./rcnn_models/voc_2007/bbox_regressor_final.mat`
  * Bounding box regressor that goes with `./rcnn_models/voc_2007/rcnn_model_finetuned`
* `./rcnn_models/voc_2012/bbox_regressor_final.mat`
  * Bounding box regressor that goes with `./rcnn_models/voc_2012/rcnn_model_finetuned`

### Caffe CNNs
* `./caffe_nets/ilsvrc_2012_train_iter_310k`
  * Reference network trained for 310k iterations (~66 epochs) on ILSVRC 2012 train
* `./caffe_nets/finetune_voc_2007_trainval_iter_70k`
  * Fine-tuned network (initialized from `ilsvrc_2012_train_iter_310k`) trained for 70k iterations on PASCAL VOC 2007 trainval
* `./caffe_nets/finetune_voc_2012_train_iter_70k`
  * Fine-tuned network (initialized from `ilsvrc_2012_train_iter_310k`) that was trained for 70k iterations on PASCAL VOC 2012 **train**

### CNN fine-tuning notes
* crop mode: warp
* crop padding: 16
* foreground sampling bias: 0.25
* foreground overlap threshold: 0.5
* background overlap threshold: 0.5
* batch size: 128
* base learning rate: 0.001
* window files: see below

### Window files used for fine-tuning
* `./window_files/*`
* Not included; these can be generated with `rcnn_make_window_file.m`.

### Pre-computed Selective Search boxes
* `./selective_search_boxes/voc_2007_train.mat`
* `./selective_search_boxes/voc_2007_val.mat`
* `./selective_search_boxes/voc_2007_trainval.mat`
* `./selective_search_boxes/voc_2007_test.mat`
* `./selective_search_boxes/voc_2012_train.mat`
* `./selective_search_boxes/voc_2012_val.mat`
* `./selective_search_boxes/voc_2012_trainval.mat`
* `./selective_search_boxes/voc_2012_test.mat`

### Selective Search notes
* The [IJCV code](http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1) was used
* For PASCAL VOC 2007, the pre-computed boxes from the website above were used
* For PASCAL VOC 2010-2012, we computed our own boxes
* "fast mode", which results in about 2k boxes per image, was used
