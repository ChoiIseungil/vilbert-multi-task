# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import torch (duplicate)
# from torch.utils.data import Dataset (duplicate)

# import json (duplicate)
# import os (duplicate)
import h5py

# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
"""
vilbert base data loader 
"""

import json
from typing import Any, Dict, List
import random
import os
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_annotations(annotations_jsonpath):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""

    annotations_json = []
    for line in open(annotations_jsonpath): annotations_json.append(json.loads(line))
    # annotations_json = json.load(open(annotations_jsonpath,encoding = "utf-8"))

    # Build an index which maps image id with a list of caption annotations.
    entries = []
    # print(f"# # # # annotations_json # # # # \n{annotations_json}")

    for annotation in annotations_json: #annotations
        # print(f"# # # # WTF annotation : {annotation}")
        entries.append(
            {
                "caption": annotation["caption"].lower(), # "caption"
                "context": annotation["context"].lower(),
                "image_id": annotation["image_id"],
            }
        )
        
    return entries


class ContextCaptionDataset(Dataset): #FoilClassificationDataset
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        features_h5path1, #: image_features_reader(ImageFeaturesH5Reader),
        features_h5path2, #: ImageFeaturesH5Reader,
        tokenizer,
        bert_model,
        padding_index=0,
        context_max_seq_length=150,
        caption_max_seq_length=30, # add
        max_region_num=96,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        
        image_features_reader = {}
        if features_h5path1 != "":
            image_features_reader[features_h5path1] = ImageFeaturesH5Reader(
                features_h5path1, False)
                
        # self._faetures_h5path1 = features_h5path1
        
        self.split = split
            
        self._image_features_reader = image_features_reader[features_h5path1]
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._context_max_seq_length = context_max_seq_length
        self._caption_max_seq_length = caption_max_seq_length # add
        self._max_region_num = max_region_num
        # self.num_labels = 2
        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(context_max_seq_length)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(context_max_seq_length) + ".pkl",
            )

        if not os.path.exists(cache_path):
            self._entries = _load_annotations(annotations_jsonpath)
            self.tokenize() # make token
            self.tensorize() # make them as indices 
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self): 
        # raw text -> encode as indices
        
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:      
            # context 
                # Note that contexts will be inputs to the encoder(vilbert)
            context_tokens = self._tokenizer.encode(entry["context"])
            context_tokens = context_tokens[: self._context_max_seq_length-2] # - cls, sep 
            context_tokens = self._tokenizer.add_special_tokens_single_sentence(context_tokens)

            context_segment_ids = [0]*len(context_tokens)
            context_input_mask = [1]*len(context_tokens)

            if len(context_tokens) < self._context_max_seq_length : 
                # pad in front of the sentence ?? 
                padding = [self._padding_index] * (self._context_max_seq_length - len(context_tokens))
                context_tokens = context_tokens + padding 
                context_input_mask += padding 
                context_segment_ids += padding 
            
            assert_eq(len(context_tokens), self._context_max_seq_length)
            entry["context_token"] = context_tokens
            entry["context_input_mask"] = context_input_mask
            entry["context_segment_ids"] = context_segment_ids

            # caption 
                # Note that captions will be inputs to a decoder 
                # # # # # for caption (as a target) # # # # # --> encoded (as index)
            caption_tokens = self._tokenizer.encode(entry["caption"])
            caption_tokens = caption_tokens[: self._caption_max_seq_length - 2] 
            caption_tokens = self._tokenizer.add_special_tokens_single_sentence(caption_tokens) # add [101, ,,,, , 102]

            caption_segment_ids = [0]*len(caption_tokens)
            caption_input_mask = [1]*len(caption_tokens)
            
            if len(caption_tokens) < self._caption_max_seq_length:
                caption_padding = [self._padding_index] * (self._caption_max_seq_length - len(caption_tokens))
                caption_tokens = caption_tokens + caption_padding
                caption_input_mask += caption_padding 
                caption_segment_ids += caption_padding
                
            assert_eq(len(caption_tokens), self._caption_max_seq_length)
            entry["caption_token"] = caption_tokens
            entry["caption_input_mask"] = caption_input_mask
            entry["caption_segment_ids"] = caption_segment_ids

    def tensorize(self): 
        # it makes just torch(numpy arr)

        for entry in self._entries:
            context_token = torch.from_numpy(np.array(entry["context_token"]))
            entry["context_token"] = context_token

            context_input_mask = torch.from_numpy(np.array(entry["context_input_mask"]))
            entry["context_input_mask"] = context_input_mask

            context_segment_ids = torch.from_numpy(np.array(entry["context_segment_ids"]))
            entry["context_segment_ids"] = context_segment_ids

            # add
            caption_token = torch.from_numpy(np.array(entry["caption_token"]))
            entry["caption_token"] = caption_token

    def __getitem__(self, index):

        entry = self._entries[index]
        image_id = entry["image_id"]

        features, num_boxes, boxes, _ = self._image_features_reader[image_id]
        
        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        features = torch.tensor(features).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(boxes).float()
        co_attention_mask = torch.zeros((self._max_region_num, self._context_max_seq_length))

        context = entry["context_token"]
        caption = entry["caption_token"] # target = 
        context_input_mask = entry["context_input_mask"]
        context_segment_ids = entry["context_segment_ids"]
        
        # caplen 
        caplen = torch.LongTensor([len(caption)])
        
        #TODO: 이해하기
        # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        # self.cpi = 1
        # all_captions = torch.LongTensor(
        #     target[((index // self.cpi) * self.cpi):(((index // self.cpi) * self.cpi) + self.cpi)])

        if index == 0 : 
            print(f"############ LOADER example############ ")
            print(f"context : {context}")
            print(f"caption : {caption}")
            print(f"features : {features}")
            print(f"caplen : {caplen}")
        
        return (
                features,
                spatials,
                image_mask,
                context,
                caption,
                context_input_mask,
                context_segment_ids,
                co_attention_mask,
                image_id,
                caplen,
                #all_captions,
            )
    def __len__(self):
        # print(f" N of total SAMPLES (from custom dataset script) : {len(self._entries)}")
        return len(self._entries)





class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size