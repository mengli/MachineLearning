#!/bin/bash

result_dir="/usr/local/google/home/limeng/githome/tensorflow/output/segnet_camvid"
output_dir="/usr/local/google/home/limeng/githome/tensorflow/output/segnet_camvid/result"

#result_dir="/usr/local/google/home/limeng/githome/tensorflow/output/segnet_kitti"
#output_dir="/usr/local/google/home/limeng/githome/tensorflow/output/segnet_kitti/result"

echo "Merge output"

train_file_names=($(ls -v $result_dir/train_*.png))
output_file_names=($(ls -v $result_dir/decision_*.png))

output_data_size=${#train_file_names[@]}

for (( i=0; i<${output_data_size}; i++ ));
do
  convert ${output_file_names[$i]} ${train_file_names[$i]} +append $output_dir/frame_$i.png
done



