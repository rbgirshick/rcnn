## R-CNN: *Regions with Convolutional Neural Network Features*

Created by Ross Girshick, Jeff Donahue, Trevor Darrell and Jitendra Malik at UC Berkeley EECS.

### Introduction

R-CNN is a state-of-the-art visual object detection system that combines bottom-up region proposals with rich features computed by a convolutional neural network. At the time of its release, R-CNN improved the previous best detection performance on PASCAL VOC 2012 by 30% relative, going from 40.9% to 53.3% mean average precision. Unlike the previous best results, R-CNN achieves this performance without using contextual rescoring or an ensemble of feature types.

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

### Installing R-CNN

0. **Prerequisites:** MATLAB (tested with 2012b on 64-bit Linux) and the Caffe [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)
1. Download the R-CNN package, which includes a version of [Caffe](http://caffe.berkeleyvision.org) that is known to work with the precomputed models (see below)
  * *Optional:* if you're feeling brave, you can clone the R-CNN git repo and the Caffe git repo and set them up manually.
2. Extract the R-CNN package: `$ tar zxvf r-cnn-release1.tgz` (this extracts safely into a folder called `rcnn`).
3. Build Caffe (this is the most complicated part)
  1. Change directories `$ cd rcnn/caffe`.
  2. Follow the instructions to build Caffe [here](http://caffe.berkeleyvision.org/installation.html).
  3. **Important:** Make sure to compile the Caffe MATLAB wrapper, which is not built by default: `$ make matcaffe`.
4. Build R-CNN
  1. Change directories back to the R-CNN root: `$ cd ..`
  2. Start MATLAB: `$ matlab -nodesktop` (or however you like to start MATLAB in your environment)
  3. Check that Caffe and MATLAB wrapper are setup correctly (this code should run without error):
  
      `>> key = caffe('get_init_key'); assert(key == -2)`
  
  3. Build R-CNN: `>> rcnn_build` (builds [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) and [Selective Search](http://www.science.uva.nl/research/publications/2013/UijlingsIJCV2013/))
4. Download the data package, which includes precompute models (see below).

### Downloading precomputed models (the data package)

The quickest way to get started is to [download precomputed R-CNN detectors](http://link/to/something). Currently we have detectors trained on PASCAL VOC 2007 and 2012 train+val. Unfortunately the download is large, so brew some coffee or take a walk while waiting.

After downloading the data package, you'll have a file called `r-cnn-release1-data.tgz`. Follow these instructions to install it:

1. Change to where you installed R-CNN: `$ cd rcnn`.
2. Extract the data package: `$ tar zxvf r-cnn-release1-data.tgz` (this will extract into a folder called `data`).

### Running an R-CNN detector on an image

Let's assume that you've downloaded the precomputed detectors. Now:

1. Change to where you installed R-CNN: `$ cd rcnn`. 
2. Start MATLAB `$ matlab -desktop`.
  * **Important:** if you don't see the message `R-CNN startup done` when MATLAB starts, then you probably didn't start MATLAB in `rcnn` directory.
3. Run the demo: `>> rcnn_demo`

### Training your own R-CNN detector on PASCAL VOC

Let's use PASCAL VOC 2007 as an example. The basic pipeline is: extract features to disk -> train SVMs -> test. You'll need about 200GB of disk space free for the feature cache (which is stored in `rcnn/feat_cache` by default; symlink `rcnn/feat_cache` elsewhere if needed). It's best if the features cache is on a fast, local disk. Before running the pipeline, we first need to install the PASCAL VOC 2007 dataset.

#### Installing PASCAL VOC 2007

1. Download the training and validation data [here](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar).
2. Download the test data [here](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar).
3. Download the VOCdevkit [here](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCdevkit_08-Jun-2007.tar).
4. Extract all of these tars into one directory, let's call it `VOCdevkit`. It should have this basic structure:

<pre>
VOCdevkit/                           % development kit
VOCdevkit/VOCcode/                   % VOC utility code
VOCdevkit/VOC2007                    % image sets, annotations, etc.
... and several other directories ...
</pre>

I use a symlink to hook the R-CNN codebase to the PASCAL VOC dataset:

`$ cd rcnn/datasets` and then `$ ln -s /your/path/to/voc2007/VOCdevkit VOCdevkit2007`

#### Extracting features

<pre>
>> rcnn_exp_cache_features('train');   % chunk1
>> rcnn_exp_cache_features('val');     % chunk2
>> rcnn_exp_cache_features('test_1');  % chunk3
>> rcnn_exp_cache_features('test_2');  % chunk4
</pre>

**Pro tip:** on a machine with one hefty GPU (e.g., k20, k40, titan) and a six-core processor, I run start two MATLAB sessions each with a three worker matlabpool. I then run chunk1 and chunk2 in parallel on that machine. In this setup, completing chunk1 and chunk2 takes about 8-9 hours (depending on your CPU/GPU combo and disk) on a single machine. Obviously, if you have more machines you can hack this function to split the workload.

#### Training R-CNN models and testing

<pre>
>> test_results = rcnn_exp_train_and_test()
</pre>

### Training an R-CNN detector on another dataset

It should be easy to train an R-CNN detector using another detection dataset as long as that dataset has *complete* bounding box annotations (i.e., all instances of all classes are labeled).

To support a new dataset, you define three functions: (1) one that returns a structure that describes the class labels and list of images; (2) one that returns a region of interest (roi) structure that describes the bounding box annotations; and (3) one that provides an test evaluation function.

You can follow the PASCAL VOC implementation as your guide:

* `imdb/imdb_from_voc.m   (list of images and classes)`  
* `imdb/roidb_from_voc.m (region of interest database)`
* `imdb/imdb_eval_voc.m   (evalutation)`  

### Fine-tuning a CNN for detection with Caffe

**TODO:** write me
