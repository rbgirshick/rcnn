#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

echo "Downloading Selective Search IJCV code..."

wget http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip

echo "Unzipping..."

unzip -q SelectiveSearchCodeIJCV.zip && rm -f SelectiveSearchCodeIJCV.zip

echo "Done."
