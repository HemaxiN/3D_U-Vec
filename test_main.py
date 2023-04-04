from test_utils import *
from tifffile import imread, imwrite
import os

model_path = r'/content/drive/My Drive/3dvectors/models/model20.hdf5' # trained model path 
save_dir = r'/content/drive/My Drive/3dvectors/results' # directory to save the images with the predicted vectors
img_dir = r'/content/drive/My Drive/3dvectors/original_images' # directory with the test images (saved as RGB .tif files)

# test parameters
_patch_size = 256 #patch size along x and y directions
_z_size = 64 #patch size along z direction
_step = 64 #overlap along x and y directions between consecutive patches extracted from the image


# non-maximum suppression thresholds
_threshold = 30
_size = 3.5

test_3duvec(model_path, img_dir, _patch_size, _z_size, _step, _threshold, _size, save_dir)

