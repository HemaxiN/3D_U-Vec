from keras import backend as K
from tensorflow.python.ops import *
import tensorflow as tf
import math
from functools import partial
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU
from keras.optimizers import Adam
K.set_image_data_format("channels_last")
try:
        from keras.engine import merge
except ImportError:
        from keras.layers.merge import concatenate
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras 
from keras.losses import categorical_crossentropy
from keras import layers as L
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from skimage import data
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import keras
from batchgenerators.augmentations.spatial_transformations import *
from math import *
from scipy.spatial import distance
from tifffile import imread, imwrite

def mse(y_true, y_pred, sample_weight=None):
    squared  = math_ops.square(y_pred - y_true)
    if sample_weight==None:
        return tf.reduce_mean(squared)
    else:
        multiplication = math_ops.multiply(sample_weight, squared)
        return tf.reduce_mean(multiplication)

def mean_se(y_true, y_pred):
    [weight1, vecxgt, vecygt, veczgt] = tf.unstack(y_true, 4, axis=4)
    [vecx, vecy, vecz] = tf.unstack(y_pred, 3, axis=4)
    vecx = tf.expand_dims(vecx, -1)
    vecxgt = tf.expand_dims(vecxgt, -1)
    vecy = tf.expand_dims(vecy, -1)
    vecygt = tf.expand_dims(vecygt, -1)
    vecz = tf.expand_dims(vecz, -1)
    veczgt = tf.expand_dims(veczgt, -1)
    vecx = K.flatten(vecx)
    vecxgt = K.flatten(vecxgt)
    vecy = K.flatten(vecy)
    vecygt = K.flatten(vecygt)
    vecz = K.flatten(vecz)
    veczgt = K.flatten(veczgt)
    epe_loss_channelx = epe_loss(vecx, vecxgt)
    epe_loss_channely = epe_loss(vecy, vecygt)
    epe_loss_channelz = epe_loss(vecz, veczgt)
    return 0.33*epe_loss_channelx + 0.33*epe_loss_channely + 0.33*epe_loss_channelz

def epe_loss(y_true, y_pred, weight):
        output = mse(y_true, y_pred, sample_weight = None)
        return output

def epe_loss1(y_true, y_pred, weight):
        output = mse(y_true, y_pred, sample_weight = weight)
        return output

def weighted_joint_loss_function(y_true, y_pred):
    [weight1, vecxgt, vecygt, veczgt] = tf.unstack(y_true, 4, axis=4)
    [vecx, vecy, vecz] = tf.unstack(y_pred, 3, axis=4)
    weight1 = tf.expand_dims(weight1, -1)
    vecx = tf.expand_dims(vecx, -1)
    vecy = tf.expand_dims(vecy, -1)
    vecz = tf.expand_dims(vecz, -1)
    vecxgt = tf.expand_dims(vecxgt, -1)
    vecygt = tf.expand_dims(vecygt, -1)
    veczgt = tf.expand_dims(veczgt, -1)
    mse_vectorsx = epe_loss1(vecxgt, vecx, weight1)
    mse_vectorsy = epe_loss1(vecygt, vecy, weight1)
    mse_vectorsz = epe_loss1(veczgt, vecz, weight1)
    return 0.33*mse_vectorsx + 0.33*mse_vectorsy + 0.33*mse_vectorsz + (1e-11)*(K.sum(K.abs(vecx))+K.sum(K.abs(vecy))+K.sum(K.abs(vecz)))


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects={'mean_se': mean_se, 'mse': mse, 'epe_loss': epe_loss,
                    'epe_loss1': epe_loss1, 'weighted_joint_loss_function': weighted_joint_loss_function}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file,custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error


def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=get_callbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience))


def Bresenham3D(x1, y1, z1, x2, y2, z2): 
    ListOfPoints = [] 
    ListOfPoints.append((x1, y1, z1)) 
    dx = abs(x2 - x1) 
    dy = abs(y2 - y1) 
    dz = abs(z2 - z1) 
    if (x2 > x1): 
        xs = 1
    else: 
        xs = -1
    if (y2 > y1): 
        ys = 1
    else: 
        ys = -1
    if (z2 > z1): 
        zs = 1
    else: 
        zs = -1
  
    # Driving axis is X-axis" 
    if (dx >= dy and dx >= dz):         
        p1 = 2 * dy - dx 
        p2 = 2 * dz - dx 
        while (x1 != x2): 
            x1 += xs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dx 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dx 
            p1 += 2 * dy 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Y-axis" 
    elif (dy >= dx and dy >= dz):        
        p1 = 2 * dx - dy 
        p2 = 2 * dz - dy 
        while (y1 != y2): 
            y1 += ys 
            if (p1 >= 0): 
                x1 += xs 
                p1 -= 2 * dy 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dy 
            p1 += 2 * dx 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Z-axis" 
    else:         
        p1 = 2 * dy - dz 
        p2 = 2 * dx - dz 
        while (z1 != z2): 
            z1 += zs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dz 
            if (p2 >= 0): 
                x1 += xs 
                p2 -= 2 * dz 
            p1 += 2 * dy 
            p2 += 2 * dx 
            ListOfPoints.append((x1, y1, z1)) 
    return ListOfPoints 




 
def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def measure_distance(x,y):
    length1 = square_rooted(x)
    length2 = square_rooted(y)
    return abs(length1-length2)

def nonmaxsuppresion(nuclei_centroids_pred, vector_directions_pred, golgi_centroids_pred, threshold_, size_):
    
    idxs = np.arange(0,len(nuclei_centroids_pred))
    #create an "idxs" list with the indexes of list vectors through which we need
    #to perform the "search" 
    
    #go through each element (i) in that list and compute the distance to all the other 
    #elements, if the distance to an element (j) is smaller than a threshold then:
    #if the length of the vector i is bigger than the length of the vector j then:
    #keep vector i as the "best" and "supress" element j (delete it from the idx list)
    #else keep the vector j as the "best" and "supress" element i and element j (delete
    #them from the idx list)
    
    #when this search finished we must obtain 1 vector and save it in the list "final_vectors"
    # initialize the list of picked indexes
    pick = []    
    while len(idxs)>0:

        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]

        if square_rooted(vector_directions_pred[i])<size_:
          suppress = [i]
          idxs = np.setdiff1d(idxs, suppress)

        else: #if the first vector has length>size_, compare with all the other idxs
          suppress = [i]
          for j in idxs:
              if j!=i:
                  
                  dist = distance.euclidean(nuclei_centroids_pred[i], nuclei_centroids_pred[j])
                  
                  if dist <= threshold_:
                      size_i = square_rooted(vector_directions_pred[i])
                      size_j = square_rooted(vector_directions_pred[j])
                      
                      if size_i >= size_j:
                          suppress.append(j)
                      elif size_j>size_:
                          suppress.append(j)
                          i = j
                      else:  
                          suppress.append(j)

          
          #vector that was picked
          pick.append(i)

          idxs = np.setdiff1d(idxs, suppress)

    
    if False:
    
        new_n_centroids_pred =  []
        new_v_directions_pred =[]
        new_g_centroids_pred = []       # will be an empty vector 
        for index in pick:
            new_n_centroids_pred.append(nuclei_centroids_pred[index])
            new_v_directions_pred.append(vector_directions_pred[index])
            
        
    else:
        new_n_centroids_pred =  []
        new_v_directions_pred = []
        new_g_centroids_pred = []
        
        for index in pick:
            new_n_centroids_pred.append(nuclei_centroids_pred[index])
            new_v_directions_pred.append(vector_directions_pred[index])
            new_g_centroids_pred.append(golgi_centroids_pred[index])
            
    return new_n_centroids_pred, new_v_directions_pred, new_g_centroids_pred


def test_3duvec(model_path, img_dir, _patch_size, _z_size, _step, _threshold, _size, save_dir):

    model = load_old_model(model_path)

    for image_nb in os.listdir(img_dir):

        image = imread(os.path.join(img_dir, image_nb))

        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        auz_sizes_or = [size_y, size_x]

        #patch size
        new_size_y = int((size_y/_patch_size) + 1) * _patch_size
        new_size_x = int((size_x/_patch_size) + 1) * _patch_size

        aux_sizes = [new_size_y, new_size_x]
        
        ## zero padding
        aux_img = np.zeros((aux_sizes[0], aux_sizes[1], np.shape(image)[2],3))
        aux_img[0:aux_sizes_or[0], 0:aux_sizes_or[1],:,:] = image
        image = aux_img

        prediction_img = np.zeros((np.shape(image)[0], np.shape(image)[1], _z_size, 3)) #x,y,z,c

        nuclei_centroids = []
        golgi_centroids = []
        vecs_pred= []
        k=0
        i =0
        while i+_patch_size <= image.shape[0]:
            j = 0
            while j+_patch_size <= image.shape[1]:

                _slice = image[i:i+_patch_size, j:j+_patch_size, :,:]
        
                _slice = _slice/255.0
                images_train = _slice

                images_train_aux = np.zeros((np.shape(images_train)[0], np.shape(images_train)[1], _z_size, 2))
                

                tstimage = np.expand_dims(images_train_aux, axis=0)
                preds_test = model.predict(tstimage)
                pred_patch = preds_test[0,:,:,:,:]


                aux0 = np.array(np.where(pred_patch[:,:,:,0]!=0)).T
                aux1 = np.array(np.where(pred_patch[:,:,:,1]!=0)).T
                aux2 = np.array(np.where(pred_patch[:,:,:,2]!=0)).T
                max_pos = np.argmax(np.asarray([len(aux0), len(aux1), len(aux2)]))
                if max_pos == 0:
                    a = aux0 
                elif max_pos == 1:
                    a = aux1
                else:
                    a = aux2
                
                for v in a:
                    if(pred_patch[v[0],v[1],v[2],0] != pred_patch[1,1,1,0] or pred_patch[v[0],v[1],v[2],1] != pred_patch[1,1,1,1] or pred_patch[v[0],v[1],v[2],2] != pred_patch[1,1,1,2]):
                        vx = pred_patch[v[0],v[1],v[2],0] 
                        vy = pred_patch[v[0],v[1],v[2],1]
                        vz = pred_patch[v[0],v[1],v[2],2] 


                        if v[0]>0+20 and v[1]>0+20 and v[0]<128-20 and v[1]<128-20:
                        #if True:

                          if np.abs(vx)>=3.5 or np.abs(vy)>=3.5:
                            if(prediction_img[i+v[0],j+v[1],v[2],0]==0 and prediction_img[i+v[0],j+v[1],v[2],1]==0 and prediction_img[i+v[0],j+v[1],v[2],2]==0):
                                prediction_img[i+v[0], j+v[1], v[2],:] = pred_patch[v[0], v[1], v[2],:]
                                nuclei_centroids.append([i+v[0],j+v[1],v[2]])
                                golgi_centroids.append([i+v[0]+vy,j+v[1]+vx,v[2]+vz])
                                vecs_pred.append([vx,vy,vz])
                                
                            else:
                                vec_existing = prediction_img[i+v[0], j+v[1], v[2], :]
                                
                                size_i = square_rooted(vec_existing)
                                size_j = square_rooted([vx,vy,vz])
                                
                                if size_j>size_i:
                                    prediction_img[i+v[0], j+v[1], v[2],:] = pred_patch[v[0], v[1], v[2],:]
                                    nuclei_centroids.append([i+v[0],j+v[1],v[2]])
                                    golgi_centroids.append([i+v[0]+vy,j+v[1]+vx,v[2]+vz])
                                    vecs_pred.append([vx,vy,vz])              
                
                k = k+1
                j = j+_step       
            i = i+_step

    
    nuclei_centroids_pred, vector_directions_pred, golgi_centroids_pred = nonmaxsuppresion(nuclei_centroids, vecs_pred, golgi_centroids, _threshold, _size)
    draw_vecs(img_dir, image_nb, save_dir, nuclei_centroids_pred, vector_directions_pred, golgi_centroids_pred)




def draw_vecs(img_dir, image_nb, save_dir, nuclei_centroids_pred, vector_directions_pred, golgi_centroids_pred):

    image = imread(os.path.join(img_dir, image_nb))

    #image size
    size_y = np.shape(image)[0]
    size_x = np.shape(image)[1]
    auz_sizes_or = [size_y, size_x]

    #patch size
    new_size_y = int((size_y/_patch_size) + 1) * _patch_size
    new_size_x = int((size_x/_patch_size) + 1) * _patch_size

    aux_sizes = [new_size_y, new_size_x]
    
    ## zero padding
    aux_img = np.zeros((aux_sizes[0], aux_sizes[1], np.shape(image)[2],3))
    aux_img[0:aux_sizes_or[0], 0:aux_sizes_or[1],:,:] = image
    image = aux_img


    image = image/255.0
    for j in range(0, len(vector_directions_pred)):
        
        n = nuclei_centroids_pred[j]
        g = golgi_centroids_pred[j]
        
        g = np.array(g)
        n = np.array(n)
        
        g = np.round(g)
        n = np.round(n)
        
        g = g.astype('int64')
        n = n.astype('int64')
        
        first_x = n[0]
        second_x = g[0]
        
        first_y = n[1]
        second_y = g[1]
        
        first_z = n[2]
        second_z = g[2]
        
        ListOfPoints = Bresenham3D(first_x, first_y, first_z, second_x, second_y, second_z)
        
        for k in range(0, len(ListOfPoints)):
            coordinates = ListOfPoints[k]
            if (coordinates[0]<image.shape[0] and coordinates[1]<image.shape[1] and coordinates[2]<56):
                image[coordinates[0], coordinates[1], coordinates[2], 2] = 1

    image = image*255.0
    image = image.astype('uint8')
    image = image[0:aux_sizes_or[0], 0:aux_sizes_or[1],:,:]

    imwrite(os.path.join(save_dir, image_nb), image, photometric='rgb')

    np.save(os.path.join(save_dir, 'nuclei_centroids' + image_nb.replace('.tif', '.npy')), nuclei_centroids_pred)
    np.save(os.path.join(save_dir, 'golgi_centroids' + image_nb.replace('.tif', '.npy')), golgi_centroids_pred)
    np.save(os.path.join(save_dir, 'vector_directions' + image_nb.replace('.tif', '.npy')), vector_directions_pred)