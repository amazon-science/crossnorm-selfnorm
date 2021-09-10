# Segmentation (Domain Generalization)

### Introduction

This repository is a PyTorch implementation of CrossNorm (CN) and SelfNorm (SN) for semantic segmentation in domain generalization setting (GTA5 -> Cityscapes). We implement CN/SN based on the scene parsing codebase (https://github.com/hszhao/semseg)


### Usage

#### Requirement:

   - Hardware: 4-8 GPUs (better with >=11G GPU memory)
   - Software: PyTorch>=1.1.0, Python3, [tensorboardX](https://github.com/lanpa/tensorboardX), 


#### Train:

   - Download GTAV and Cityscapes datasets and symlink the paths to them as follows (you can alternatively modify the relevant paths specified in folder `config`):

     ```
     cd segmentation
     mkdir -p dataset
     ln -s /path_to_gtav_dataset dataset/gtav
     ```

   - Download ImageNet pre-trained models and put them under folder `initmodel` for weight initialization. [ResNet50-SN](https://drive.google.com/file/d/1a_u67UuSZJUhQ-4DGMMrSRXQIWdzcBa2/view?usp=sharing) are available.

   - Specify the gpu used in config then do training:

     ```shell
     sh tool/train_cnsn.sh gtav fcn50_cnsn
     ```
      
    - For other detailed usage, please refer to the origin repo: https://github.com/hszhao/semseg


