#!/bin/bash

file_prefixes=(
	imdb_5
	imdb_6
	imdb_7
	)


for file_prefix in ${file_prefixes[@]}; do
    wget https://emrbucket-dag20180305.s3.amazonaws.com/ds1004-project/raw/${file_prefix}.tar
    tar -xvf ${file_prefix}.tar
    python3 make_smaller_sparse_matrices.py imdb imdb.mat imdb output_file_${file_prefix}.npz
    rm -rf imdb
    rm ${file_prefix}.tar
done

