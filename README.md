# TaoismNet:A Fruiter Segmentation Model Based on MinimalismDesign for UAV camera

<img src="https://github.com/MRonaldo-gif/TaoismNet/blob/main/TaoismNet/Technical%20roadmap%20and%20workflow..png" width="1100">

This repository is the official implementation of the following paper:

> **TaoismNet:A Fruiter Segmentation Model Based on Minimalism**<br>
> [Yanheng Mai](https://github.com/MRonaldo-gif), [Jiaqi Zheng](https://github.com/kidous2333), [Zefeng Luo](), [Jianqiang Lu](),[Chaoran Yu](),[Caili Yu](),[Zhongliang Liao]()

> 
> > **Abstract**<br>
> > <font size=3> *The development of precision agriculture requires UAV to collect diverse data, such as
RGB images, 3D point clouds, and hyperspectral images.Recently,convolutional networks have
made remarkable progress in downstream visual tasks,ignoring the contradiction between accuracy
and speed in the UAVs segmentation. The study aims to provide further valuable insights into
efficient model named Taoism-Net.The acheivements indicate the following:(1) Prescription maps in
agricultural UAVs requires pixel level precise segmentation, most of works focused on accurancy
but not latency simultaneously, being uncapable of satisfying the expectations of practical tasks.(2)
Taoism-Net is a refreshingly segment model, overcoming the challenges of complexity in deep
learning, based on minimalist design, which is used to generate prescription maps through pixel level
classification mapping of geodetic coordinates (The lychee tree aerial dataset in Guangdong is used for
experiments).(3) Compared with mainstream lightweight models or mature segmentation algorithms,
Taoism-Net has achieved significant improvements, increased by at least 4.8% in mIoU,and manifested
superior performance in the accuracy-latency curve.(4)“The greatest truths is concise” was widely
spreaded by ancient Taoism,indicating that the most fundamental approach is reflected through the
utmost minimalism,moreover,Taoism-Net expects to build bridge between academic research and
industrial deployment,for example UAVs in precision agriculture.*</font>



## Getting Started

- This repository is based on the mmsegmentation of [OpenMMLab](https://openmmlab.com/). Follow the [requirements](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements) of it before the steps below. 
- Clone the repository:
    ```shell
    git clone https://github.com/MRonaldo-gif/TaoismNet.git
    cd TaoismNet
    ```
    
## Dataset

- To make PASCAL_VOC2012 preprocess the dataset images and json into one files as the input_dir,output_dir is where the reasult dataset will be saved,
    ```shell
    python tolls/labelme2voc.py <input_dir> <output_dir> --labels<the txt file of labels>
    ```
    The txt file of labels include three line text like(you can add your class below last line):
- \_\_ignore__
- \_background_
- tree
## File Addition

- Add or replace previous documents to subsequent documents
- 1: TaoismNet/registry --> mmseg
- 2: TaoismNet/backbones\\__init__.py --> mmseg\models\backbones
- 3: TaoismNet/backbones/TaoismNet.py --> mmseg\models\backbones
- 4: TaoismNet/evaluation --> mmseg
## Pretrain

- Pretrain a TaoismNet model on VOC dataset
    ```shell
    python tools/train.py configs/swin-tiny.py --workdir <the dir to save logs and models> 
    
    # You may also try different values for the following settings
    # --resume: resume from the latest checkpoint in the work_dir automatically
    # --amp: enable automatic-mixed-precision training
  ```

## Test
- Test the model on a given dataset
    ```shell
    python tools/test.py configs/swin-tiny.py <check_point> --workdir <the dir to save logs and models>
 
    # You may also try different values for the following settings
    # --out: The directory to save output prediction for offline evaluation
    # --show: show prediction results
    # --show-dir: directory where painted images will be saved
    ```
## Inference: Defect Image Generation

- Use the model to inference
    ```shell
    # you can used the tool to get the binary image
    python tools/PREDICT.py --config --checkpoint --img --device --save_dir
 
    # You may also try different values for the following settings
    # --config:Config file
    # --checkpoint:Checkpoint file
    # --img:Image file
    # --device:device
    # --save_dir:the reasult
   
    # you can used the tool to get the mask rendering
    python tools/image_demo.py --config --checkpoint --img --device --save_dir
 
    # You may also try different values for the following settings
    # --config:Config file
    # --checkpoint:Checkpoint file
    # --img:Image file
    # --device:device
    # --save_dir:the reasult
    ```
  



## Acknowledgements

- The work was supported by the Key-Area Research and DevelopmentProgram of Guangdong Province(2023B0202090001);Key Research and Development Program of Guangzhou(2023B03J1392); The National Natural Science Foundation of China (42061046); The special projects in
key fields of ordinary universities in Guangdong Province (2021ZDZX4111) .
- This repository have used codes from [OpenMMLab](https://openmmlab.com/) and [mmsegmentation]([https://github.com/richzhang/PerceptualSimilarity](https://github.com/open-mmlab/mmsegmentation)).
