#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

echo "Downloading the R-CNN data package (precomputed models)..."

wget http://www.cs.berkeley.edu/~rbg/r-cnn-release1-data.tgz

echo "Unzipping..."

tar zxvf r-cnn-release1-data.tgz && rm -f r-cnn-release1-data.tgz

echo "Done."
