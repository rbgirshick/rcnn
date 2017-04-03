#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

FILE=r-cnn-release1-selective-search.tgz
CHECKSUM=6cf6df219c1e514f64482f11d00bd0b4

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Model checksum is correct. No need to download."
    exit 0
  else
    echo "Model checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading precomputed selective search boxes (1.8G)..."

wget https://dl.dropboxusercontent.com/s/uf2i1y2oee7c6n1/r-cnn-release1-selective-search.tgz

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
