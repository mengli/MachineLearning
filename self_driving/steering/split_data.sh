#!/bin/bash
# Split Nvida dataset into train data and test data

src_dir="driving_dataset"
train_dst_dir="train_data"
test_dst_dir="test_data"

train_data_size=40000
data_size=45568

echo "Split Nvida driving dataset into train data and test data"

rm -rf $train_dst_dir $test_dst_dir
mkdir $train_dst_dir $test_dst_dir

# train_data
i=0
while [ $i -lt $train_data_size ]
do
  cp $src_dir/"$i.jpg" $train_dst_dir/"$i.jpg"
  true $(( i++ ))
done

# test_data
while [ $i -lt $data_size ]
do
  cp $src_dir/"$i.jpg" $test_dst_dir/"$i.jpg"
  true $(( i++ ))
done

