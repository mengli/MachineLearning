#!/bin/bash

test_data_dir="/usr/local/google/home/limeng/Downloads/kitti/data_road/testing/image_2"
fake_label_data_dir="/usr/local/google/home/limeng/Downloads/kitti/data_road/testing"

echo "KITTI test dataset"

rm -f test.txt
touch test.txt

test_file_names=($(ls $test_data_dir))
test_data_size=${#test_file_names[@]}

for (( i=0; i<${test_data_size}; i++ ));
do
    echo $test_data_dir/${test_file_names[$i]} $fake_label_data_dir/umm_road_000000.png >> test.txt
done
