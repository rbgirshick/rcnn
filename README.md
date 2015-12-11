## R-CNN: *Regions with Convolutional Neural Network Features*

Created by Ross Girshick, Jeff Donahue, Trevor Darrell and Jitendra Malik at UC Berkeley EECS.

Acknowledgements: a huge thanks to Yangqing Jia for creating Caffe and the BVLC team, with a special shoutout to Evan Shelhamer, for maintaining Caffe and helping to merge the R-CNN fine-tuning code into Caffe.

### Introduction

R-CNN is a state-of-the-art visual object detection system that combines bottom-up region proposals with rich features computed by a convolutional neural network. At the time of its release, R-CNN improved the previous best detection performance on PASCAL VOC 2012 by 30% relative, going from 40.9% to 53.3% mean average precision. Unlike the previous best results, R-CNN achieves this performance without using contextual rescoring or an ensemble of feature types.

R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1311.2524) and will appear in a forthcoming CVPR 2014 paper.

### Citing R-CNN

If you find R-CNN useful in your research, please consider citing:

    @inproceedings{girshick14CVPR,
        Author = {Girshick, Ross and Donahue, Jeff and Darrell, Trevor and Malik, Jitendra},
        Title = {Rich feature hierarchies for accurate object detection and semantic segmentation},
        Booktitle = {Computer Vision and Pattern Recognition},
        Year = {2014}
    }

### License

R-CNN is released under the Simplified BSD License (refer to the
LICENSE file for details).

### PASCAL VOC detection results

Method         | VOC 2007 mAP | VOC 2010 mAP | VOC 2012 mAP
-------------- |:------------:|:------------:|:------------:
R-CNN          | 54.2%        | 50.2%        | 49.6%
R-CNN bbox reg | 58.5%        | 53.7%        | 53.3%

* VOC 2007 per-class results are available in our [CVPR14 paper](http://www.cs.berkeley.edu/~rbg/#girshick2014rcnn)
* VOC 2010 per-class results are available on the [VOC 2010 leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb_dt.php?challengeid=6&compid=4)
* VOC 2012 per-class results are available on the [VOC 2012 leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb_dt.php?challengeid=11&compid=4)
* These models are available in the model package (see below)

### ImageNet 200-class detection results

Method         | ILSVRC2013 test mAP 
---------------|:-------------------:
R-CNN bbox reg | 31.4%

* For more details see the updated [R-CNN tech report](http://arxiv.org/abs/1311.2524v3) (Sections 2.5 and 4, in particular)
* This model is available in the model package (see below)
* The code that was used for training is in the `ilsvrc` branch (still needs some cleanup before merging into `master`)

### Installing R-CNN

0. **Prerequisites** 
  0. MATLAB (tested with 2012b on 64-bit Linux)
  0. Caffe's [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)
0. **Install Caffe** (this is the most complicated part)
  0. R-CNN has been checked for compatability against Caffe release v0.999. *It has not been updated to work with the current Caffe master.*
  0. Download [Caffe v0.999](https://github.com/BVLC/caffe/archive/v0.999.tar.gz)
  0. Follow the [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)
  0. Let's call the place where you installed caffe `$CAFFE_ROOT` (you can run `export CAFFE_ROOT=$(pwd)`)
  0. **Important:** Make sure to compile the Caffe MATLAB wrapper, which is not built by default: `make matcaffe`
  1. **Important:** Make sure to run `cd $CAFFE_ROOT/data/ilsvrc12 && ./get_ilsvrc_aux.sh` to download the ImageNet image mean
0. **Install R-CNN**
  0. Get the R-CNN source code by cloning the repository: `git clone https://github.com/rbgirshick/rcnn.git`
  0. Now change into the R-CNN source code directory: `cd rcnn`
  0. R-CNN expects to find Caffe in `external/caffe`, so create a symlink: `ln -sf $CAFFE_ROOT external/caffe`
  0. Start MATLAB (make sure you're still in the `rcnn` directory): `matlab`
  0. You'll be prompted to download the [Selective Search](http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1) code, which we cannot redistribute. Afterwards, you should see the message `R-CNN startup done` followed by the MATLAB prompt `>>`.
  1. if you see `Warning: Please install Caffe in ./external/caffe` most probably instead of folder `caffe` you have `+caffe`, so create another symlink to solve this warning. `ln -sf $CAFFE_ROOT/caffe/matlab/+caffe  $CAFFE_ROOT/caffe/matlab/caffe`.
  0. Run the build script: `>> rcnn_build()` (builds [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) and [Selective Search](http://www.science.uva.nl/research/publications/2013/UijlingsIJCV2013/)). Don't worry if you see compiler warnings while building liblinear, this is normal on my system.
  0. Check that Caffe and MATLAB wrapper are set up correctly (this code should run without error): `>> key = caffe('get_init_key');` (expected output is key = -2)
  1. if you get the error `Undefined function or variable 'caffe'`, most probably you haven't done step 1.
  0. Download the model package, which includes precompute models (see below).

**Common issues:** You may need to set an `LD_LIBRARY_PATH` before you start MATLAB. If you see a message like "Invalid MEX-file '/path/to/rcnn/external/caffe/matlab/caffe/caffe.mexa64': libmkl_rt.so: cannot open shared object file: No such file or directory" then make sure that CUDA and MKL are in your `LD_LIBRARY_PATH`. On my system, I use:

    export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
  

### Downloading pre-computed models (the model package)

The quickest way to get started is to download pre-computed R-CNN detectors. Currently we have detectors trained on PASCAL VOC 2007 train+val, 2012 train, and ILSVRC13 train+val. Unfortunately the download is large (1.5GB), so brew some coffee or take a walk while waiting.

From the `rcnn` folder, run the model fetch script: `./data/fetch_models.sh`. 

This will populate the `rcnn/data` folder with `caffe_nets` and `rcnn_models`. See `rcnn/data/README.md` for details.

Pre-computed selective search boxes can also be downloaded for VOC2007, VOC2012, and ILSVRC13.
From the `rcnn` folder, run the selective search data fetch script: `./data/fetch_selective_search_data.sh`.

This will populate the `rcnn/data` folder with `selective_selective_data`.

**Caffe compatibility note:** R-CNN has been updated to use the new Caffe proto messages that were rolled out in Caffe v0.999. The model package contains models in the up-to-date proto format. If, for some reason, you need to get the old (Caffe proto v0) models, they can still be downloaded: [VOC models](http://www.cs.berkeley.edu/~rbg/r-cnn-release1-data-caffe-proto-v0.tgz) 
 [ILSVRC13 model](http://www.cs.berkeley.edu/~rbg/r-cnn-release1-data-ilsvrc2013-caffe-proto-v0.tgz).

### Running an R-CNN detector on an image

Let's assume that you've downloaded the precomputed detectors. Now:

1. Change to where you installed R-CNN: `cd rcnn`. 
2. Start MATLAB `matlab`.
  * **Important:** if you don't see the message `R-CNN startup done` when MATLAB starts, then you probably didn't start MATLAB in `rcnn` directory.
3. Run the demo: `>> rcnn_demo`
3. Enjoy the detected bicycle and person

### Training your own R-CNN detector on PASCAL VOC

Let's use PASCAL VOC 2007 as an example. The basic pipeline is: 

    extract features to disk -> train SVMs -> test
    
You'll need about 200GB of disk space free for the feature cache (which is stored in `rcnn/feat_cache` by default; symlink `rcnn/feat_cache` elsewhere if needed). **It's best if the feature cache is on a fast, local disk.** Before running the pipeline, we first need to install the PASCAL VOC 2007 dataset.

#### Installing PASCAL VOC 2007

0. Download the training, validation, test data and VOCdevkit:

  <pre>
  wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  </pre>

0. Extract all of these tars into one directory, it's called `VOCdevkit`. 

  <pre>
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_08-Jun-2007.tar
  </pre>

0. It should have this basic structure:

  <pre>
  VOCdevkit/                           % development kit
  VOCdevkit/VOCcode/                   % VOC utility code
  VOCdevkit/VOC2007                    % image sets, annotations, etc.
  ... and several other directories ...
  </pre>

0. I use a symlink to hook the R-CNN codebase to the PASCAL VOC dataset:

  <pre>
  ln -sf /your/path/to/voc2007/VOCdevkit /path/to/rcnn/datasets/VOCdevkit2007
  </pre>

#### Extracting features

<pre>
>> rcnn_exp_cache_features('train');   % chunk1
>> rcnn_exp_cache_features('val');     % chunk2
>> rcnn_exp_cache_features('test_1');  % chunk3
>> rcnn_exp_cache_features('test_2');  % chunk4
</pre>

**Pro tip:** on a machine with one hefty GPU (e.g., k20, k40, titan) and a six-core processor, I run start two MATLAB sessions each with a three worker matlabpool. I then run chunk1 and chunk2 in parallel on that machine. In this setup, completing chunk1 and chunk2 takes about 8-9 hours (depending on your CPU/GPU combo and disk) on a single machine. Obviously, if you have more machines you can hack this function to split the workload.

#### Training R-CNN models and testing

Now to run the training and testing code, use the following experiments script:

<pre>
>> test_results = rcnn_exp_train_and_test()
</pre>

**Note:** The training and testing procedures save models and results under `rcnn/cachedir` by default. You can customize this by creating a local config file named `rcnn_config_local.m` and defining the experiment directory variable `EXP_DIR`. Look at `rcnn_config_local.example.m` for an example.


### Training an R-CNN detector on another dataset

It should be easy to train an R-CNN detector using another detection dataset as long as that dataset has *complete* bounding box annotations (i.e., all instances of all classes are labeled).

To support a new dataset, you define three functions: (1) one that returns a structure that describes the class labels and list of images; (2) one that returns a region of interest (roi) structure that describes the bounding box annotations; and (3) one that provides an test evaluation function.

You can follow the PASCAL VOC implementation as your guide:

* `imdb/imdb_from_voc.m   (list of images and classes)`  
* `imdb/roidb_from_voc.m (region of interest database)`
* `imdb/imdb_eval_voc.m   (evalutation)`  

### Fine-tuning a CNN for detection with Caffe

As an example, let's see how you would fine-tune a CNN for detection on PASCAL VOC 2012.

0. Create window files for VOC 2012 train and VOC 2012 val.
  0. Start MATLAB in the `rcnn` directory
  0. Get the imdb for VOC 2012 train: `>> imdb_train = imdb_from_voc('datasets/VOCdevkit2012', 'train', '2012');`
  0. Get the imdb for VOC 2012 val: `>> imdb_val = imdb_from_voc('datasets/VOCdevkit2012', 'val', '2012');`
  0. Create the window file for VOC 2012 train: `>> rcnn_make_window_file(imdb_train, 'external/caffe/examples/pascal-finetuning');`
  0. Create the window file for VOC 2012 val: `>> rcnn_make_window_file(imdb_val, 'external/caffe/examples/pascal-finetuning');`
  0. Exit MATLAB
0. Run fine-tuning with Caffe
  0. Copy the fine-tuning prototxt files: `cp finetuning/voc_2012_prototxt/pascal_finetune_* external/caffe/examples/pascal-finetuning/`
  0. Change directories to `external/caffe/examples/pascal-finetuning`
  0. Execute the fine-tuning code (make sure to replace `/path/to/rcnn` with the actual path to where R-CNN is installed):
  
  <pre>
  GLOG_logtostderr=1 ../../build/tools/finetune_net.bin \
  pascal_finetune_solver.prototxt \
  /path/to/rcnn/data/caffe_nets/ilsvrc_2012_train_iter_310k 2>&1 | tee log.txt
  </pre>
      
**Note:** In my experiments, I've let fine-tuning run for 70k iterations, although with hindsight it appears that improvement in mAP saturates at around 40k iterations.
