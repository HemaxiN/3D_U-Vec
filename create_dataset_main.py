import numpy as np
import os
from tifffile import imread
from create_dataset_utils import *

imgs_dir = '/mnt/2TBData/hemaxi/3dvectors/dataset3d/256_rc/train/images' #tif files, rgb images (256,256,64,3)
vecs_dir = '/mnt/2TBData/hemaxi/3dvectors/dataset3d/256_rc/train/gt' #np array, (size Nx6), N is the number of vectors
                                                                     #in the patch, ((x,y,z) positions of the nucleus 
                                                                     #centroid) and (vx, vy, vz) are the components 
                                                                     #of the nucleus-Golgi vector

save_dir_img = '/dev/shm/3dvectors/'
save_dir_vec = '/dev/shm/3dvectors/'

_sizee = 256
_zsize = 64

npatches = 109 #number of patches in "imgs_dir"
maxpatches = 2000 #number of augmented patches

k=0
ii=0
while k<maxpatches:

    final_img = np.zeros((_sizee,_sizee,_zsize,2))
    if ii==npatches:
        ii = 0

    img_aux = imread(os.path.join(imgs_dir, str(ii) + '.tif'))
    msk_aux = np.load(os.path.join(vecs_dir, str(ii) + '.npy'))
    img_aux = img_aux/255.0

    final_vectors = np.zeros(np.shape(msk_aux))
    
    #data augmentation
    if(np.random.choice([0,1])==1):
        img_aux, msk_aux = vertical_flip(img_aux, msk_aux)
    if(np.random.choice([0,1])==1):
        img_aux, msk_aux = horizontal_flip(img_aux, msk_aux)
    if(np.random.choice([0,1])==1):
        img_aux, msk_aux = intensity(img_aux, msk_aux)
    if(np.random.choice([0,1])==1):
        angle = np.random.choice(np.arange(0,360,90))
        img_aux, msk_aux = rotation(img_aux, msk_aux, angle)

    final_img[0:_sizee,0:_sizee,0:size_z_dir,:] = img_aux[:,:,0:size_z_dir,0:2]
    final_vectors = msk_aux
      
    np.save(os.path.join(save_dir_vec, str(k) + '.npy'), final_vectors)
    np.save(os.path.join(save_dir_img, str(k) + '.npy'), final_img)
    k=k+1
    ii = ii + 1
