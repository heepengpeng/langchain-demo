"""
Text detection and recognition inference

Example:
    $ python tools/infer/ocr_infer/predict_system.py --image_dir {path_to_img_file} --det_algorithm DB++ \
      --rec_algorithm CRNN
    $ python tools/infer/ocr_infer/predict_system.py --image_dir {path_to_img_dir} --det_algorithm DB++ \
      --rec_algorithm CRNN_CH
"""
import argparse
import json
import logging
import os
import sys
from time import time
from typing import Union

import cv2
import mindspore as ms
import numpy as np

from .predict_det import TextDetector
from .predict_rec import TextRecognizer
from .utils import crop_text_region, get_image_paths

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from .mindocr.utils.logger import set_logger
from .mindocr.utils.visualize import visualize  # noqa

logger = logging.getLogger("mindocr")


class TextSystem(object):
    def __init__(self, args):
        self.text_detect = TextDetector(args)
        self.text_recognize = TextRecognizer(args)

        self.box_type = args.det_box_type
        self.drop_score = args.drop_score
        self.save_crop_res = args.save_crop_res
        self.crop_res_save_dir = args.crop_res_save_dir
        if self.save_crop_res:
            os.makedirs(self.crop_res_save_dir, exist_ok=True)
        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        self.vis_font_path = args.vis_font_path

    def __call__(self, img_or_path: Union[str, np.ndarray], do_visualize=True):
        """
        Detect and recognize texts in an image

        Args:
            img_or_path (str or np.ndarray): path to image or image rgb values as a numpy array

        Return:
            boxes (list): detected ocr_infer boxes, in shape [num_boxes, num_points, 2], where the point coordinate (x, y)
                follows: x - horizontal (image width direction), y - vertical (image height)
            texts (list[tuple]): list of (ocr_infer, score) where ocr_infer is the recognized ocr_infer string for each box,
                and score is the confidence score.
            time_profile (dict): record the time cost for each sub-task.
        """
        assert isinstance(img_or_path, str) or isinstance(
            img_or_path, np.ndarray
        ), "Input must be string of path to the image or numpy array of the image rgb values."
        fn = os.path.basename(img_or_path).split(".")[0] if isinstance(img_or_path, str) else "img"

        time_profile = {}
        start = time()

        # detect ocr_infer regions on an image
        det_res, data = self.text_detect(img_or_path, do_visualize=False)
        time_profile["det"] = time() - start
        polys = det_res["polys"].copy()
        logger.info(f"Num detected ocr_infer boxes: {len(polys)}\nDet time: {time_profile['det']}")

        # crop ocr_infer regions
        crops = []
        for i in range(len(polys)):
            poly = polys[i].astype(np.float32)
            cropped_img = crop_text_region(data["image_ori"], poly, box_type=self.box_type)
            crops.append(cropped_img)

            if self.save_crop_res:
                cv2.imwrite(os.path.join(self.crop_res_save_dir, f"{fn}_crop_{i}.jpg"), cropped_img)
        # show_imgs(crops, is_bgr_img=False)

        # recognize cropped images
        rs = time()
        rec_res_all_crops = self.text_recognize(crops, do_visualize=False)
        time_profile["rec"] = time() - rs

        # logger.info(
        #     "Recognized texts: \n"
        #     + "\n".join([f"{text}\t{score}" for text, score in rec_res_all_crops])
        #     + f"\nRec time: {time_profile['rec']}"
        # )

        # filter out low-score texts and merge detection and recognition results
        boxes, text_scores = [], []
        for i in range(len(polys)):
            box = det_res["polys"][i]
            # box_score = det_res["scores"][i]
            text = rec_res_all_crops[i][0]
            text_score = rec_res_all_crops[i][1]
            if text_score >= self.drop_score:
                boxes.append(box)
                text_scores.append((text, text_score))

        time_profile["all"] = time() - start

        # visualize the overall result
        if do_visualize:
            vst = time()
            vis_fp = os.path.join(self.vis_dir, fn + "_res.png")
            # TODO: improve vis for leaning texts
            visualize(
                data["image_ori"],
                boxes,
                texts=[x[0] for x in text_scores],
                vis_font_path=self.vis_font_path,
                display=False,
                save_path=vis_fp,
                draw_texts_on_blank_page=False,
            )  # NOTE: set as you want
            time_profile["vis"] = time() - vst
        return boxes, text_scores, time_profile


def get_infer_res(boxes_all, text_scores_all, img_paths):
    for i, img_path in enumerate(img_paths):
        # fn = os.path.basename(img_path).split('.')[0]
        boxes = boxes_all[i]
        text_scores = text_scores_all[i]

        res = []  # result for current image
        for j in range(len(boxes)):
            res.append(
                {
                    "transcription": text_scores[j][0],
                    "points": np.array(boxes[j]).astype(np.int32).tolist(),
                }
            )

        return res


def extrac_text(infer_res):
    result = ""
    all_sorted_list = []
    data = infer_res
    sorted_data = sorted(data, key=lambda x: x["points"][0][1])

    len_width = 30
    sub_r = []
    for index in range(len(sorted_data) - 1):
        sub_r.append(sorted_data[index])
        if abs(sorted_data[index + 1]["points"][0][1] - sorted_data[index]["points"][0][1]) > len_width:
            all_sorted_list.append(sub_r)
            sub_r = []
    for sorted_item in all_sorted_list:
        transcription = ""
        sorted_item = sorted(sorted_item, key=lambda x: x["points"][0][0])
        for item in sorted_item:
            transcription += item["transcription"]
        result += f"{transcription}\n"
    return result


def main(image_path):
    args = argparse.Namespace()

    args.mode = 0
    args.image_dir = os.path.abspath(image_path)

    args.det_algorithm = "DB++"
    args.rec_rec_algorithm = "CRNN_CH"

    args.det_model_dir = ""
    args.det_amp_level = "O0"
    args.det_limit_type = "max"
    args.det_box_type = "quad"
    args.det_db_thresh = 0.3
    args.det_db_box_thresh = 0.6
    args.det_db_unclip_ratio = 1.5
    args.max_batch_size = 10
    args.use_dilation = False
    args.det_db_score_mode = "fast"

    args.rec_algorithm = "RARE_CH"
    args.rec_image_shape = "3, 32, 320"
    args.rec_batch_mode = True
    args.rec_batch_num = 8
    args.max_text_length = 25
    args.rec_score_thresh = 0.3
    args.warmup = False
    args.rec_model_dir = ""
    args.rec_amp_level = "O0"
    args.drop_score = 0.5
    args.det_limit_side_len = 960
    args.draw_img_save_dir = "./inference_result"
    args.crop_res_save_dir = "./output"
    args.vis_font_path = "docs/fonts/simfang.ttf"
    args.visualize_output = False
    args.save_crop_res = False

    set_logger(name="mindocr")
    img_paths = get_image_paths(args.image_dir)
    ms.set_context(mode=args.mode)

    # init ocr_infer system with detector and recognizer
    text_spot = TextSystem(args)

    # warmup
    if args.warmup:
        for i in range(2):
            text_spot(img_paths[0], do_visualize=False)

    boxes_all, text_scores_all = [], []
    for i, img_path in enumerate(img_paths):
        boxes, text_scores, time_prof = text_spot(img_path, do_visualize=args.visualize_output)
        boxes_all.append(boxes)
        text_scores_all.append(text_scores)
    infer_res = get_infer_res(boxes_all, text_scores_all, img_paths)
    ocr_text = extrac_text(infer_res)
    return ocr_text
