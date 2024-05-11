import os
import torch
import cv2
import argparse
import numpy as np
from pprint import pprint
from tqdm import tqdm
from mmseg.apis import init_model, inference_model

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('--img',  help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--device', help='device')
    parser.add_argument('--save_dir', help='save_dir')

    args = parser.parse_args()
    return args


def make_full_path(root_list, root_path):
    file_full_path_list = []
    for filename in root_list:
        file_full_path = os.path.join(root_path, filename)
        file_full_path_list.append(file_full_path)
    return file_full_path_list


def read_filepath(root):
    from natsort import natsorted
    test_image_list = natsorted(os.listdir(root))
    test_image_full_path_list = make_full_path(test_image_list, root)
    return test_image_full_path_list


def main():
    args = parse_args()

    model_mmseg = init_model(args.config, args.checkpoint, device=args.device)

    for imgs in tqdm(read_filepath(args.img)):
        result = inference_model(model_mmseg, imgs)
        pred_mask = result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        pred_mask[pred_mask == 1] = 255
        save_path = os.path.join(args.save_dir, f"{os.path.basename(args.config).split('.')[0]}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, f"{os.path.basename(result.img_path).split('.')[0]}.png"), pred_mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("hello")


main()
