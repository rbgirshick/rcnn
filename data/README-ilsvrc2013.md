## Contents -- Precomputed R-CNN Models

These models are now downloaded using the fetch_data.sh script. Note that the code to train ilsvrc13 models has not been merged into master yet. It is available in the ilsvrc13 branch. However, the trained models are compatible with the master branch.

### R-CNN detectors trained on ILSVRC2013
* `./rcnn_models/ilsvrc2013/rcnn_model.mat`
  * Uses `./caffe_nets/finetune_ilsvrc13_val1+train1k_iter_50000`
  * This detector was trained on val (i.e. val1+val2) plus up to 1000 images per category from train
* `./rcnn_models/ilsvrc2013/bbox_regressor_final.mat`
  * Bounding-box regressor that goes with `./rcnn_models/ilsvrc2013/rcnn_model.mat`


### Caffe CNN
* `./caffe_nets/finetune_ilsvrc13_val1+train1k_iter_50000`
  * CNN fine-tuned on val1 plus up to 1000 images per category from train
  * initialized from `./caffe_nets/ilsvrc_2012_train_iter_310k` in the main R-CNN data package

### Selective search boxes on val1, val2, val, and test
(Downloaded using data/fetch_selective_search_data.sh)
./selective_search_data/ilsvrc13_val1.mat
./selective_search_data/ilsvrc13_val2.mat
./selective_search_data/ilsvrc13_test.mat
./selective_search_data/ilsvrc13_val.mat


### Image lists in the val1/val2 split (blacklisted images are pre-filtered)
./splits/val1.txt
./splits/val2.txt
