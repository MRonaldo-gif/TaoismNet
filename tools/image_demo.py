# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
IMAGE_FILE_PATH = r"/home/user4/zjq/mmsegmentation/images/0215.jpg"
# 模型训练结果的config配置文件路径
CONFIG = r'/home/user4/zjq/mmsegmentation/tools/work_dirs/swin-tiny/swin-tiny.py'
# 模型训练结果的权重文件路径
CHECKPOINT = r'/home/user4/zjq/mmsegmentation/tools/work_dirs/swin-tiny/V13.pth'
# 模型推理测试结果的保存路径，每个模型的推理结果都保存在`{save_dir}/{模型config同名文件夹}`下，如文末图片所示。
SAVE_DIR = r"/home/user4/zjq/mmsegmentation/tools/PREDICT/0215.jpg"

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument('--out-file', default=None, help='Path to output file')

    parser.add_argument('--img', default=IMAGE_FILE_PATH, help='Image file')
    parser.add_argument('--config', default=CONFIG, help='Config file')
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='Checkpoint file')

    parser.add_argument('--out-file', default=SAVE_DIR, help='save_dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
    print("hello")