# /usr/bin/bash

mkdir __run__
cd __run__
wget https://zenodo.org/records/4277620/files/models.tar.gz?download=1
tar -xvzf models.tar.gz
rm -rf models.tar.gz
