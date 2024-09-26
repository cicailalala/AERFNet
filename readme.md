# [ICASSP2024] Hybrid Module with Multiple Receptive Fields and Self-Attention Layers for Medical Image Segmentation

This repository is the official implementation of **[Hybrid Module with Multiple Receptive Fields and Self-Attention Layers for Medical Image Segmentation](https://ieeexplore.ieee.org/document/10445854), ICASSP2024**. We test all **13 organs** in the Synapse dataset instead of 8 organs, which is typically reported in previous works. Our implementation refers to the implementation of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [UNet-2022](https://github.com/282857341/UNet-2022). 

# Table of contents  
- [Installation](#Installation) 
- [Data Preparation](#Data_Preparation)
- [Data Preprocessing](#Data_Preprocessing)
- [Training and Testing](#Training_and_Testing) 
- [Experiment Results](#Experiment_Results) 
# Installation
```
git clone https://github.com/cicailalala/AERFNet.git
cd AERFNet
conda env create -f environment.yml
source activate AERFNet
pip install -e .
```
# Data_Preparation
Following [UNet-2022](https://github.com/282857341/UNet-2022), our proposed model is a 2D based network, and all data should be expressed in 2D form with ```.nii.gz``` format. You can download the organized dataset from the [link](https://drive.google.com/drive/folders/1b4IVd9pOCFwpwoqfnVpsKZ6b3vfBNL6x?usp=sharing) or download the original data from the link below. If you need to convert other formats (such as ```.jpg```) to the ```.nii.gz```, you can look up the file and modify the [file](https://github.com/282857341/UNet-2022/blob/master/nnunet/dataset_conversion/Task120_ISIC.py) based on your own datasets.

**Dataset I**
[ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/) consists of 100 MRI scans. For each sample, the myocardium (MYO), right ventricle (RV), and left ventricle (LV) are labeled for segmentation. In our experiment, we set the ratio of training, validation and testing to 7:1:2.

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) contains 3779 axial abdominal clinical slices extracted from 30 CT scans. Following common practice, 18 samples were split for training and 12 samples for testing. The average Dice-Similarity coefficient (DSC) and average Hausdorff Distance (HD) are adopted as evaluation metrics.

**Dataset III**
[EM dataset](https://imagej.net/events/isbi-2012-segmentation-challenge#training-data) is a neural structures segmentation dataset that involves 30 images with the size of 512*512. The training dataset contains 24 samples while the validation and test dataset contains 3 and 3 samples, repectively.

The dataset should be finally organized as follows:
```
./nnUNet_data/
  ├── nnUNet_raw/
      ├── nnUNet_raw_data/
          ├── Task05_Synapse13/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
              ├── evaulate.py              
          ......
      ├── nnUNet_cropped_data/
  ├── nnUNet_preprocessed/
      ├── Task005_Synapse13/
  ├── nnUNet_trained_models/
```
# Data_Preprocessing
```
nnUNet_convert_decathlon_task -i path/to/nnUNet_raw_data/Task01_ACDC
```
This step will convert the name of folder from Task01 to Task001, and make the name of each nifti files end with '_000x.nii.gz'.
```
nnUNet_plan_and_preprocess -t 1
```
Where ```-t 1``` means the command will preprocess the data of the Task001_ACDC.
Before this step, you should set the environment variables to ensure the framework could know the path of ```nnUNet_raw```, ```nnUNet_preprocessed```, and ```nnUNet_trained_models```. 
The detailed construction can be found in [UNet-2022](https://github.com/282857341/UNet-2022) and [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)

You can download the processed Synapse dataset (13 organs) from the Baidu Cloud [link](https://pan.baidu.com/s/1tGjMERRhxyCTeZAaEZgGIw), password: v8xg.

# Training_and_Testing
```
bash train_or_test.sh -c 0 -n aerfnet_synapse13_320 -i 5 -s 0.2 -t true -p true 
```
- ```-c 0``` refers to the index of your Cuda device and this command only support the single-GPU training.
- ```-n aerfnet_synapse13_320``` denotes the suffix of the trainer located at ```AERFNet/nnunet/training/network_training/```. For example, nnUNetTrainerV2_aerfnet_synapse13_320 refers to ```-n aerfnet_synapse13_320```.
- ```-i 5``` means the index of the task. For example, Task005 refers to ```-i 5```.
- ```-s 0.2``` means the inference step size, reducing the value tends to bring better performance but longer inference time.
- ```-t true/false``` determines whether to run the training command.
- ```-p true/false``` determines whether to run the testing command.

The above command will run the training command and testing command continuously.

Before you start the testing, please make sure the model_best.model and model_best.model.pkl exists in the specified path, like this:
```
nnUNet_trained_models/nnUNet/2d/Task005_synapse13/nnUNetTrainerV2_aerfnet_synapse13_320/fold_0/model_best.model
nnUNet_trained_models/nnUNet/2d/Task005_synapse13/nnUNetTrainerV2_aerfnet_synapse13_320/fold_0/model_best.model.pkl
```
# Experiment_Results
Results on **Synapse dataset** on **all 13 organs**. (Spleen, Right/Left Kidney, Gallbladder, Esophagus, Liver, Stomach, Aorta, Inferior Vena Cava, Portal Vein Splenic Vein, Pancreas and Right/Left Adrenal Gland)
| **Method**  | **Dimension** | **DSC** | **HD** | **Splee** | **RKid** | **LKid** | **Gall** | **Eso** | **Liver** | **Sto** | **Aorta** | **IVC** | **Veins** | **Pancr** | **RAd** | **LAd** |
|-----------------------------------|-------------------------------|-------------------------------|--------------------------------------|------------------------|-----------------------|-----------------------|-----------------------|----------------------|------------------------|----------------------|------------------------|----------------------|------------------------|------------------------|----------------------|----------------------|
| [TransUNet](https://arxiv.org/abs/1409.1556)    | 2D                            | 70.17                                | 24.73                  | 84.78                 | 69.04                 | 76.09                 | 48.64                | 68.04                  | 94.24                | 73.51                  | 86.93                | 77.29                  | 65.61                  | 61.81                | 56.14                | 50.06             |
| [SwinUNet](https://arxiv.org/abs/1409.1556)          | 2D                            | 66.65                                | 24.01                  | 86.49                 | 72.62                 | 76.02                 | 55.14                | 63.22                  | 93.07                | 69.25                  | 80.84                | 66.92                  | 55.49                  | 52.17                | 52.39                | 42.87             |
| [MT-UNet](https://arxiv.org/abs/1409.1556)             | 2D                            | 71.43                                | 22.40                  | 87.86                 | 74.93                 | 82.38                 | 57.91                | 73.64                  | 94.47                | 65.33                  | 86.22                | 75.09                  | 62.80                  | 55.52                | 60.03                | 52.39             |
| [MISSFormer](https://arxiv.org/abs/1409.1556)      | 2D                            | 76.87                                | 11.51                  | 87.81                 | 82.57                 | 86.31                 | 65.95                | 74.80                  | 95.72                | 79.13                  | 89.32                | 80.71                  | 69.29                  | 67.01                | 58.98                | 61.71             |
| [nnUNet](https://arxiv.org/abs/1409.1556)              | 2D                            | 79.92                                | 19.01                  | 90.09                 | 82.01                 | 87.16                 | 65.91                | 77.35                  | *96.21*    | 79.84                  | *91.28*    | *85.22*      | 71.94                  | 73.98                | **69.39**       | 68.62             |
| [UNet-2022](https://arxiv.org/abs/1409.1556)         | 2D                            | *80.29*                    | 16.28                  | 89.65                 | 85.66                 | *88.40*     | 58.91                | 76.58                  | 96.02                | 82.33                  | 90.91                | 84.70                  | *75.20*      | *77.81*    | 67.80                | 69.80             |
| [V-Net](https://arxiv.org/abs/1409.1556)       | 3D                            | 73.37                                | 18.94                  | 85.81                 | 83.00                 | 82.08                 | 60.06                | 67.79                  | 95.09                | 76.99                  | 85.49                | 81.16                  | 65.25                  | 63.50                | 50.11                | 57.47             |
| [UNETR](https://arxiv.org/abs/1409.1556) | 3D                            | 73.39                                | 15.81                  | 88.24                 | 83.77                 | 83.25                 | 62.89                | 70.69                  | 95.43                | 74.68                  | 85.63                | 77.83                  | 61.62                  | 59.42                | 61.48                | 49.14             |
| [SwinUNETR](https://arxiv.org/abs/1409.1556)    | 3D                            | 77.04                                | 32.00                  | 86.61                 | 85.28                 | 85.12                 | **68.63**       | 75.84                  | 91.72                | 76.77                  | 89.09                | 83.81                  | 70.18                  | 65.03                | 60.10                | 63.33             |
| [nnUNet](https://arxiv.org/abs/1409.1556)             | 3D                            | 78.46                                | *11.07*      | *91.17*     | *86.20*     | 87.40                 | 63.92                | 75.31                  | 95.52                | 71.73                  | 89.91                | 83.76                  | 70.54                  | 70.16                | 68.84                | 65.53             |
| [nnFormer](https://arxiv.org/abs/1409.1556)  | 3D                            | 80.24                                | 16.27                  | 88.79                 | 84.90                 | 83.23                 | 65.90                |**79.94**         | 95.80                | *84.34*      | 89.97                | 84.07                  | 72.83                  | **78.83**       | 62.97                | **71.54**    |
| **Ours**                     | 2D                            | **82.13**                       | **10.89**         | **91.26**        | **87.81**        | **89.74**        | *66.19*    | *77.97*      | **96.48**       | **86.64**         | **91.87**       | **87.01**         | **77.76**         | 76.07                | *69.03*    | *69.89* |


## Reference
* [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* [UNet-2022](https://github.com/282857341/UNet-2022) 
* [UNeXt](https://github.com/jeya-maria-jose/UNeXt-pytorch)
## Citations

```bibtex
@inproceedings{qi2024hybrid,
  title={Hybrid Module with Multiple Receptive Fields and Self-Attention Layers for Medical Image Segmentation},
  author={Qi, Wenbo and Zhou, Wenyong and Wong, Ngai and Chan, SC},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1906--1910},
  year={2024},
  organization={IEEE}
}
```
