#!/bin/bash

file_prefixes=(
	imdb_1,
	imdb_2,
	imdb_3,
	imdb_4,
	imdb_5
	)


for file_prefix in ${file_prefixes[@]}; do
    wget https://emrbucket-dag20180305.s3.amazonaws.com/ds1004-project/raw/${file_prefixes}.tar
    tar -xvf ${file_prefix}
    python3 make_sparse_matrix.py imdb imdb.mat imdb output_file_${file_prefix}.npz
    rm -rf imdb
done





