# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Requires vqa-maskrcnn-benchmark to be built and installed. See Readme
# for more details.

"""
Usage e.g) 
./script/run_extract_features.sh
python script/extract_features.py --data GA --batch_size 4 --gpu_num 5
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import time

import sys
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

import pandas as pd
from urllib.request import urlopen

sys.path.insert(0,'../vqa-maskrcnn-benchmark/')
CSVPATH = "/mnt/nas2/seungil/refined_legacy/"
FEATUREPATH = "/mnt/nas2/seungil/features/"
class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.csvpath = CSVPATH + self.args.data + '.csv'
        self.featurepath = FEATUREPATH + self.args.data + '/'
        self.detection_model = self._build_detection_model()

        os.makedirs(self.featurepath, exist_ok=True)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_file", 
            default='data/detectron_model.pth', 
            type=str, 
            help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", 
            default='data/detectron_config.yaml', 
            type=str, 
            help="Detectron config file"
        )
        parser.add_argument(
            "--batch_size", 
            type=int, 
            default=2, 
            help="Batch size"
        )
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--data", 
            type=str, 
            help="Image directory or file"
        )
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        parser.add_argument(
            "--gpu_num", 
            type=int, 
            default=0, 
            help="A number of GPU will use"
        )
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        torch.cuda.set_device(f"cuda:{self.args.gpu_num}")

        model.to(f"cuda:{self.args.gpu_num}")
        model.eval()
        return model

    def _image_transform(self, path):
        image_url = "https:" + path['image url'] 
        if image_url.split('.')[-1] == 'gif' : 
            print(f"Oops, it's a  gif file format.")

        image_bytes = None
        while image_bytes is None:
            try:
                with urlopen(image_url) as f : 
                    image_bytes = f.read()
            except:
                print("Too many request Error T.T\n",image_url)
                time.sleep(5)
        
        encoded_img = np.fromstring(image_bytes, dtype = np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)


        im = img.astype(np.float32)


        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        else:
            if im.shape[2]>3:
                im=im[:,:,:3]
            elif im.shape[2]<3:
                im=np.concatenate((im, im[:,:,[1]]), axis=2)

        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
            cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "cls_prob": scores[keep_boxes].cpu().numpy(),
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths): # some rows are input
        img_tensor, im_scales, im_infos = [], [], []

        for df_index, row in image_paths.iterrows():
            # print(f"idx: {df_index}\ttitle: {row['title']}")
            im, im_scale, im_info = self._image_transform(row) # each row 
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to(f"cuda:{self.args.gpu_num }")

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.replace(".","")
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.featurepath, file_base_name), info)
        print(f"{file_base_name} is saved")

    def extract_features(self):
        image_dir = self.csvpath
        extension = image_dir.split('.')[-1]

        if os.path.isfile(image_dir) and extension != 'csv':
            features, infos = self.get_detectron_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        
        elif extension == 'csv' : 
            print("a CSV file is detected")
            image_df = pd.read_csv(image_dir)
            for chunk in self._chunks(image_df, self.args.batch_size): # chunk is some rows in df
                try : 
                    features, infos = self.get_detectron_features(chunk) # some rows will be input data
                    for i, (_, row) in enumerate(chunk.iterrows()): # idx have to be matched with ,,, row index 
                        file_name = str(row['Unnamed: 0']) + '_' + row['title']
                        self._save_feature(file_name, features[i], infos[i])
                except Exception as e : 
                    print(f"error : {e}")
                    continue 
        else:
            files = glob.glob(os.path.join(image_dir, "*"))
            for chunk in self._chunks(files, self.args.batch_size):
                try:
                    features, infos = self.get_detectron_features(chunk)
                    for idx, file_name in enumerate(chunk):
                        self._save_feature(file_name, features[idx], infos[idx])
                except BaseException:
                    continue


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
