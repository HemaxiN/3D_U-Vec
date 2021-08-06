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

def epe_loss1(y_true, y_pred, weight):
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


def 3DUVec(n_classes=3, im_sz=256, depth=64, n_channels=2, n_filters_start=8, growth_factor=2, upconv=True):
        droprate=0.10
        n_filters = n_filters_start
        inputs = Input((im_sz, im_sz, depth, n_channels))
        #inputs = BatchNormalization(axis=-1)(inputs)
        conv1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
        conv1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv1)
        #pool1 = Dropout(droprate)(pool1)

        n_filters *= growth_factor
        pool1 = BatchNormalization(axis=-1)(pool1)
        conv2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
        conv2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv2)
        pool2 = Dropout(droprate)(pool2)

        n_filters *= growth_factor
        pool2 = BatchNormalization(axis=-1)(pool2)
        conv3 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
        conv3 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv3)
        pool3 = Dropout(droprate)(pool3)

        n_filters *= growth_factor
        pool3 = BatchNormalization(axis=-1)(pool3)
        conv4_0 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool3)
        conv4_0 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4_0)
        pool4_1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv4_0)
        pool4_1 = Dropout(droprate)(pool4_1)

        n_filters *= growth_factor
        pool4_1 = BatchNormalization(axis=-1)(pool4_1)
        conv4_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool4_1)
        conv4_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv4_1)
        pool4_2 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(conv4_1)
        pool4_2 = Dropout(droprate)(pool4_2)

        n_filters *= growth_factor
        conv5 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(pool4_2)
        conv5 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)

        n_filters //= growth_factor
        if upconv:
                up6_1 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv5), conv4_1])
        else:
                up6_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4_1])
        up6_1 = BatchNormalization(axis=-1)(up6_1)
        conv6_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up6_1)
        conv6_1 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6_1)
        conv6_1 = Dropout(droprate)(conv6_1)

        n_filters //= growth_factor
        if upconv:
                up6_2 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv6_1), conv4_0])
        else:
                up6_2 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_1), conv4_0])
        up6_2 = BatchNormalization(axis=-1)(up6_2)
        conv6_2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up6_2)
        conv6_2 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv6_2)
        conv6_2 = Dropout(droprate)(conv6_2)

        n_filters //= growth_factor
        if upconv:
                up7 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv6_2), conv3])
        else:
                up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv3])
        up7 = BatchNormalization(axis=-1)(up7)
        conv7 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up7)
        conv7 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)
        conv7 = Dropout(droprate)(conv7)

        n_filters //= growth_factor
        if upconv:
                up8 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv7), conv2])
        else:
                up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2])
        up8 = BatchNormalization(axis=-1)(up8)
        conv8 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up8)
        conv8 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv8)
        conv8 = Dropout(droprate)(conv8)

        n_filters //= growth_factor
        if upconv:
                up9 = concatenate([Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_last')(conv8), conv1])
        else:
                up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1])
        conv9 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(up9)
        conv9 = Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same', data_format='channels_last')(conv9)

        conv11 = Conv3D(n_classes, (1, 1, 1), activation='linear', data_format='channels_last')(conv9)

        model = Model(inputs=inputs, outputs=conv11)    
        model.compile(optimizer=Adam(), loss=weighted_joint_loss_function, metrics = [mean_se])
        return model

#Learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
        return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, logging_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                                    learning_rate_patience=50, verbosity=1,
                                    early_stopping_patience=None):
        callbacks = list()
        callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
        callbacks.append(CSVLogger(logging_file, append=True))
        if learning_rate_epochs:
                callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                                                                             drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
        else:
                callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                                                                     verbose=verbosity))
        if early_stopping_patience:
                callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
        return callbacks


def load_old_model(model_file):
        print("Loading pre-trained model")
        custom_objects = {'mean_se': mean_se, 'mse':mse, 'epe_loss':epe_loss, 'epe_loss1':epe_loss1, 'weighted_joint_loss_function':weighted_joint_loss_function}
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


def train_model(model, model_file, logging_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                                learning_rate_patience=20, early_stopping_patience=None):
        model.fit_generator(generator=training_generator,
                                                steps_per_epoch=steps_per_epoch,
                                                epochs=n_epochs,
                                                validation_data=validation_generator,
                                                validation_steps=validation_steps,
                                                callbacks=get_callbacks(model_file, logging_file,
                                                                                                initial_learning_rate=initial_learning_rate,
                                                                                                learning_rate_drop=learning_rate_drop,
                                                                                                learning_rate_epochs=learning_rate_epochs,
                                                                                                learning_rate_patience=learning_rate_patience,
                                                                                                early_stopping_patience=early_stopping_patience))

# Generates data for Keras

class DataGenerator(keras.utils.Sequence):

        def __init__(self, partition, configs, data_dir, data_aug_dict=None):

                self.data_aug_dict = data_aug_dict
                self.partition = partition
                self.data_dir = data_dir
                self.list_IDs = sorted(os.listdir(self.data_dir+partition+'/images/'),key=self.order_dirs)

                self.dim = configs['dim']
                self.mask_dim = configs['mask_dim']
                self.batch_size = configs['batch_size']
                self.shuffle = configs['shuffle']
                self.on_epoch_end()

        def __len__(self):
                'Denotes the number of batches per epoch'
                return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):
                'Generate one batch of data'
                # Generate indexes of the batch
                indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

                # Find list of IDs
                list_IDs_temp = [self.list_IDs[k] for k in indexes]

                # Generate data
                X, mask = self.__data_generation(list_IDs_temp)

                if True:
                        X, mask = self.augment(X,mask)
                        X, mask = self.compute_weights(X,mask)    
                return X, mask

        def on_epoch_end(self):
                'Updates indexes after each epoch'
                self.indexes = np.arange(len(self.list_IDs))
                if self.shuffle == True:
                        np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
                # Initialization
                X = np.empty((self.batch_size, *self.dim))
                mask = []
                # Generate data
                for i, ID_path in enumerate(list_IDs_temp):
                        X[i,] = np.load(self.data_dir + self.partition +'/images/'+ ID_path)
                        mask.append(np.load(self.data_dir + self.partition +'/outputs/'+ ID_path))
                return X, mask

        def order_dirs(self, element):
                return element.replace('.npy','')

        def augment(self, X, mask):
                X = self.rescale_img_values(X)
                return X, mask

        def rescale_img_values(self,img, max=None, min=None):
                img = img/255.0
                return img

        def compute_weights(self, X, mask_gt):
                #mask: list with N length, each n contains vectors for image n
                IMG_SIZE = self.mask_dim[0]
                _depth = self.mask_dim[2]
                final_msk = np.empty((self.batch_size, *self.mask_dim))
                iii = 0
                for vec_array in mask_gt:

                            img_aux = np.zeros((IMG_SIZE,IMG_SIZE,_depth,3))

                            for v in vec_array:
                                    if(int(v[1])<IMG_SIZE and int(v[0])<IMG_SIZE and int(v[2])<_depth):       
                                            img_aux[int(v[1]), int(v[0]), int(v[2]), 0] = v[3]
                                            img_aux[int(v[1]), int(v[0]), int(v[2]), 1] = v[4]
                                            img_aux[int(v[1]), int(v[0]), int(v[2]), 2] = v[5]

                            masks_train = img_aux

                            masks_train_a = np.zeros((np.shape(masks_train)[0], np.shape(masks_train)[1], _depth, np.shape(masks_train)[3]))
                            masks_train_a[:,:,:,:] = masks_train[:,:,:,:] 
                            masks_train = masks_train_a

                            masks = masks_train

                            mask = masks[:,:,:,:]
                            
                            centroids_n = np.zeros(np.shape(mask[:,:,:,0]))
                            
                            nuclei_centroids_pred = []
                            
                            vectors_pred = masks[:,:,:,:] 
                            aux0 = np.array(np.where(vectors_pred[:,:,:,0]!=0)).T
                            aux1 = np.array(np.where(vectors_pred[:,:,:,1]!=0)).T
                            aux2 = np.array(np.where(vectors_pred[:,:,:,2]!=0)).T
                            max_pos = np.argmax(np.asarray([len(aux0), len(aux1), len(aux2)]))
                            if max_pos == 0:
                                    a = aux0 
                            elif max_pos == 1:
                                    a = aux1
                            else:
                                    a = aux2
                
                            for v in a:
                                    vx = vectors_pred[v[0],v[1],v[2], 0] 
                                    vy = vectors_pred[v[0],v[1],v[2], 1] 
                                    vz = vectors_pred[v[0],v[1],v[2], 2] 
                                    nuclei_centroids_pred.append([v[0], v[1], v[2]])

                            for centroid in nuclei_centroids_pred:
                                    if int(centroid[0])<IMG_SIZE and int(centroid[1])<IMG_SIZE:
                                            centroids_n[int(centroid[0]), int(centroid[1]), int(centroid[2])] = 1
                                    else:
                                            centroids_n[int(centroid[0])-4, int(centroid[1])-4, int(centroid[2])] = 1


                            centroids_ng = np.zeros((np.shape(centroids_n)[0],np.shape(centroids_n)[1], np.shape(centroids_n)[2],4))
                            masks_aux_aux = 5*np.ones(np.shape(centroids_n))
                            masks_aux_aux[centroids_n!=0] = 10000
                            centroids_ng[:,:,:,0] = masks_aux_aux

                            centroids_ng[:,:,:,1] = img_aux[:,:,:,0]
                            centroids_ng[:,:,:,2] = img_aux[:,:,:,1]
                            centroids_ng[:,:,:,3] = img_aux[:,:,:,2]

                            final_msk[iii,:,:,:,:] = centroids_ng

                            iii = iii + 1
                return X, final_msk
