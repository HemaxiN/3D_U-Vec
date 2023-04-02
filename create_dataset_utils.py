import cv2
import numpy as np
from scipy.ndimage.interpolation import rotate
import math

##auxiliary function to rotate the vector
def vector_rotation(x,y, angle):
    angle_rad = (np.pi*angle)/(180)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    point = np.array([y,x]).T
    [ry, rx] = rot_matrix.dot(point)
    return rx,ry

##auxiliary function to rotate the point
def rotate_around_point_lowperf(image, pointx, pointy, angle):
    """Rotate a point around a given point.
    
    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    radians = (np.pi*angle)/(180)
    x, y = pointx, pointy
    ox, oy = image.shape[0]/2, image.shape[1]/2

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy - math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy


##rotation
def rotation(image, vector, angle):
    
    rot_image = np.zeros(np.shape(image))
    for z in range(0, rot_image.shape[2]):
        rot_image[:,:,z,:] = rotate(image[:,:,z,:], angle, mode='constant', reshape=False)
        
    rot_vector = []
    for v in vector:
        rv0, rv1 = rotate_around_point_lowperf(image,v[0], v[1], angle)
        rv3, rv4 = vector_rotation(v[3], v[4], angle)
        rot_vector.append([rv0, rv1, v[2], rv3, rv4, v[5]])
    return rot_image, rot_vector


##vertical flip
def vertical_flip(image, vector):
    
    flippedimage = np.zeros(np.shape(image))
    for z in range(0, flippedimage.shape[2]):
        flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 0)
    
    flippedvec = []
    for v in vector:
        fv0, fv1, fv2 = v[0], (image.shape[1]-v[1]-1), v[2]
        fv3, fv4, fv5 = v[3], -v[4], v[5]
        flippedvec.append([fv0, fv1, fv2, fv3, fv4, fv5])
    return flippedimage, flippedvec


##horizontal flip
def horizontal_flip(image, vector):
    
    flippedimage = np.zeros(np.shape(image))
    for z in range(0, flippedimage.shape[2]):
        flippedimage[:,:,z,:] = cv2.flip(image[:,:,z,:], 1)
        
        
    flippedvec = []
    for v in vector:
        fv0, fv1, fv2 = image.shape[0]-v[0]-1, v[1], v[2]
        fv3, fv4, fv5 = -v[3], v[4], v[5]
        flippedvec.append([fv0, fv1, fv2, fv3, fv4, fv5])
    return flippedimage, flippedvec

#intensity variations
def intensity(image, vector, alpha=None):    
        image = image.astype('float64')
        image = image*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        image = image.astype('float64')
    return image, vector


