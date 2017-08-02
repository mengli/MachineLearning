#!/bin/bash

data_dir="/usr/local/google/home/limeng/Downloads/camvid/701_StillsRaw_full"
label_data_dir="/usr/local/google/home/limeng/Downloads/camvid/LabeledApproved_full/image_2"

echo "Camvid dataset"

rm -f train.txt
touch train.txt

data_file_names=($(ls $data_dir))
label_file_names=($(ls $label_data_dir))
data_size=${#data_file_names[@]}

for (( i=0; i<${data_size}; i++ ));
do
    echo $data_dir/${data_file_names[$i]} $label_data_dir/${label_file_names[$i]} >> train.txt
done
