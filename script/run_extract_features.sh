#!/bin/bash
# rm -r /mnt/nas2/seungil/features
mkdir /mnt/nas2/seungil/features

mkdir /mnt/nas2/seungil/features/FA
python script/extract_features.py --data FA --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/GA
python script/extract_features.py --data GA --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA0
python script/extract_features.py --data AA0 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA1
python script/extract_features.py --data AA1 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA2
python script/extract_features.py --data AA2 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA3
python script/extract_features.py --data AA3 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA4
python script/extract_features.py --data AA4 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA5
python script/extract_features.py --data AA5 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA6
python script/extract_features.py --data AA6 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA7
python script/extract_features.py --data AA7 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA8
python script/extract_features.py --data AA8 --batch_size 4 --gpu_num 5

mkdir /mnt/nas2/seungil/features/AA9
python script/extract_features.py --data AA9 --batch_size 4 --gpu_num 5
