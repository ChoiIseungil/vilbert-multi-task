# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Command: 
# ./script/run_convert_to_lmdb.sh
# python script/convert_to_lmdb.py --data AA1,AA2,AA3 --out train 

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm

MAP_SIZE = 1099511627776
PATH = '/mnt/nas2/seungil/'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", 
        required = True,
        default = None, 
        type = str, 
        help = "Path to extracted features file (Can read multiple files at once)"
    )
    parser.add_argument(
        "--out", 
        required = True,
        default = None, 
        type = str, 
        help = "Path to extracted features file"
    )
    return parser

def main():
    args = get_parser().parse_args()
    infiles = []
    id_list = []
    for file in args.data.split(','):
        infiles.extend(glob.glob(os.path.join(PATH + 'features/' + file, "*")))

    env = lmdb.open(PATH + 'lmdbs/' + args.out + '/', map_size=MAP_SIZE)

    with env.begin(write=True) as txn:
        for infile in tqdm.tqdm(infiles):
            reader = np.load(infile, allow_pickle=True).item()  
            item = {}
            item["image_id"] = reader.get("image_id")
            img_id = str(item["image_id"]).encode()
            id_list.append(img_id)
            item["image_h"] = reader.get("image_height")
            item["image_w"] = reader.get("image_width")
            item["num_boxes"] = reader.get("num_boxes")
            item["boxes"] = reader.get("bbox")
            item["features"] = reader.get("features")
            txn.put(img_id, pickle.dumps(item))
            
        txn.put("keys".encode(), pickle.dumps(id_list))


if __name__ == "__main__":
    main()