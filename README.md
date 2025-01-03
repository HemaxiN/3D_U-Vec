# 3D U-Vec: A Deep Learning Based Approach for the Prediction of Nucleus-Golgi Vectors in 3D Microscopy Images



This repository contains the Python implementation of the 3D U-Vec, a Deep Learning model to predict nucleus-Golgi vectors in 3D microscopy images of mouse retinas, as described in: 

- Hemaxi Narotamo, Cláudio Franco, Margarida Silveira, 3D U-Vec: Prediction of Nucleus-Golgi Vectors in 3D Microscopy Images.

If you are using this code in your research please [cite the paper](#how-to-cite).

## Architecture

Our approach uses a 3D U-Net as a neural network backbone to predict the 3D vectors from the input images.

![](https://github.com/HemaxiN/3D_U-Vec/blob/main/images/overviewa.png)

## Overview

3D U-Vec predicts, for each voxel belonging to the centroid of the nucleus, the (x,y,z) components of the corresponding polarity vector. More specifically, it receives as input a 3D patch (X,Y,Z,2), where the first and the second channels correspond to the red and green channels of the original microscopy images, respectively; and outputs a (X,Y,Z,3) tensor. Each non-zero component in the output represents the x, y and z components of each of the detected nucleus-Golgi pairs, located approximately at the centroid of the corresponding nucleus.

![](https://github.com/HemaxiN/3D_U-Vec/blob/main/images/overview.png)

## Requirements

Python 3.5.2, Tensorflow-GPU 1.9.0, Keras 2.2.4 and other packages listed in [requirements.txt](https://github.com/HemaxiN/3D_U-Vec/blob/main/requirements.txt).

## Training on your own dataset

Change the `imgs_dir`, `vecs_dir`, `save_dir_img`, `save_dir_vec`, `_xysize`, `_zsize` and `maxpatches` parameters in file [create_dataset_main.py](https://github.com/HemaxiN/3D_U-Vec/blob/main/create_dataset_main.py), where:

*  `imgs_dir`: directory containing the RGB image patches (X,Y,Z,3), saved as .tif files;
*  `vecs_dir`: directory containing the corresponding np arrays, of size Nx6, where N is the number of vectors in the corresponding patch. The first three components are the (x,y,z) positions of the nucleus centroid, and the last three components (vx,vy,vz) the components of the nucleus-Golgi vector;
*  `save_dir_img`: directory where the processed images will be saved (after performing data augmentation as described in the paper);
*  `save_dir_vec`: directory where the corresponding vectors are saved;
*  `_xysize`: size of the microscopy image patch along x and y directions;
*  `_zsize`: size of the microscopy image patch along the z direction (pre-process the image: perform zero-padding if the number of slices is smaller than 64)
*  `maxpatches`: number of augmented patches.

Run the file [create_dataset_main.py](https://github.com/HemaxiN/3D_U-Vec/blob/main/create_dataset_main.py) to create the training/validation dataset. The dataset obtained with this code should have the following tree structure:

```
train_val_dataset
├── train
│   ├── images
│   └── outputs
└── val
    ├── images
    └── outputs
```

Thereafter, change the `_size`,`_z_size`,`data_dir`,`save_dir` and other training parameters in the file [train_main.py](https://github.com/HemaxiN/3D_U-Vec/blob/main/train_main.py), where:

* `_size`: size of the microscopy image patch along x and y directions;
* `_s_size`: size of the microscopy image patch along the z direction;
* `data_dir`: directory with the structure depicted above;
* `save_dir`: directory where the models and the log file will be saved.

Run the file [train_main.py](https://github.com/HemaxiN/3D_U-Vec/blob/main/train_main.py) to train de model.

## Testing

Change the `model_path`, `save_dir`, `img_dir`, `_patch_size`, `_z_size`, `_step` parameters in the file [test_main.py](https://github.com/HemaxiN/3D_U-Vec/blob/main/test_main.py), where:

* `model_path`: path to the trained model;
* `save_dir`: directory to save the images with the predicted vectors (as .tif files) and the predicted vectors (as .npy arrays);
* `img_dir`: directory with the test images (saved as RGB .tif files);
* `_patch_size`: patch size along x and y directions;
* `_z_size`: patch size along z direction;
* `_step`: overlap along x and y directions between consecutive patches extracted from the image.

Run the file [test_main.py](https://github.com/HemaxiN/3D_U-Vec/blob/main/test_main.py).


## How to cite
```bibtex
@article{narotamo20223duvec,
  title={3D U-Vec: A Deep Learning Based Approach for the Prediction of Nucleus-Golgi Vectors in 3D Microscopy Images},
  author={Narotamo, Hemaxi and Franco, Cláudio and Silveira, Margarida},
  booktitle={},
  pages={},
  year={},
  organization={}
}
```

