#!/bin/bash

train_data_dir="/usr/local/google/home/limeng/Downloads/kitti/data_road/training"

echo "KITTI dataset"

rm -f train.txt
touch train.txt

append_data_items()
{
  train_file_names=($(ls $train_data_dir/image_2/$1))
  gt_file_names=($(ls $train_data_dir/gt_image_2/$1))

  train_data_size=${#train_file_names[@]}

  for (( i=0; i<${train_data_size}; i++ ));
  do
    echo ${train_file_names[$i]} ${gt_file_names[$i]} >> train.txt
  done
}

append_data_items "um_*"
append_data_items "umm_*"
append_data_items "uu_*"



