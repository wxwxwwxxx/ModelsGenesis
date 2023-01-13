import random
import numpy as np
from scipy import ndimage
class CutBlack3D(object):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
    def __call__(self, img):

        mask = np.sum(img,axis=-1)
        mask = np.where(mask <= self.threshold,0,1)
        # print(mask.shape)
        x_sum = mask.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)[:,0,0]
        y_sum = mask.sum(axis=0, keepdims=True).sum(axis=2, keepdims=True)[0,:,0]
        z_sum = mask.sum(axis=0, keepdims=True).sum(axis=1, keepdims=True)[0,0,:]
        # print(x_sum)
        x_range = np.where(x_sum != 0)[0]
        y_range = np.where(y_sum != 0)[0]
        z_range = np.where(z_sum != 0)[0]
        # print(x_range.shape)
        x_min = np.min(x_range)
        y_min = np.min(y_range)
        z_min = np.min(z_range)
        x_max = np.max(x_range)
        y_max = np.max(y_range)
        z_max = np.max(z_range)
        img = img[x_min:x_max,y_min:y_max,z_min:z_max,:]
        return img
class RandomAlign3D(object):
    def __init__(self, length_x,length_y,length_z):
        self.length_x = length_x
        self.length_y = length_y
        self.length_z = length_z
    def __call__(self, img):
        x, y, z, c = np.shape(img)
        #cut
        crop_x_l = random.randint(0, x - self.length_x) if x >= self.length_x else 0
        crop_y_l = random.randint(0, y - self.length_y) if y >= self.length_y else 0
        crop_z_l = random.randint(0, z - self.length_z) if z >= self.length_z else 0
        img = img[crop_x_l:crop_x_l + self.length_x, crop_y_l:crop_y_l + self.length_y, crop_z_l:crop_z_l + self.length_z, :]

        pad_x_l = random.randint(0, self.length_x - x) if x < self.length_x else 0
        pad_y_l = random.randint(0, self.length_y - y) if y < self.length_y else 0
        pad_z_l = random.randint(0, self.length_z - z) if z < self.length_z else 0
        pad_x_r = self.length_x - x - pad_x_l if x < self.length_x else 0
        pad_y_r = self.length_y - y - pad_y_l if y < self.length_y else 0
        pad_z_r = self.length_z - z - pad_z_l if z < self.length_z else 0

        img = np.pad(img, ((pad_x_l, pad_x_r), (pad_y_l, pad_y_r), (pad_z_l, pad_z_r), (0, 0)),
                     'constant',
                     constant_values=(0, 0))

        return img

class FixedAlign3D(object):
    def __init__(self, length_x,length_y,length_z):
        self.length_x = length_x
        self.length_y = length_y
        self.length_z = length_z
    def __call__(self, img):
        x, y, z, c = np.shape(img)
        crop_x_l = (x - self.length_x) // 2 if x >= self.length_x else 0
        crop_y_l = (y - self.length_y) // 2 if y >= self.length_y else 0
        crop_z_l = (z - self.length_z) // 2 if z >= self.length_z else 0
        img = img[crop_x_l:crop_x_l + self.length_x, crop_y_l:crop_y_l + self.length_y, crop_z_l:crop_z_l + self.length_z, :]

        pad_x_l = (self.length_x - x) // 2 if x < self.length_x else 0
        pad_y_l = (self.length_y - y) // 2 if y < self.length_y else 0
        pad_z_l = (self.length_z - z) //2  if z < self.length_z else 0
        pad_x_r = self.length_x - x - pad_x_l if x < self.length_x else 0
        pad_y_r = self.length_y - y - pad_y_l if y < self.length_y else 0
        pad_z_r = self.length_z - z - pad_z_l if z < self.length_z else 0

        img = np.pad(img, ((pad_x_l, pad_x_r), (pad_y_l, pad_y_r), (pad_z_l, pad_z_r), (0, 0)),
                     'constant',
                     constant_values=(0, 0))
        return img

class RandomRotation3D(object):
    def __init__(self,max_angle):
        self.max_angle = max_angle
    def __call__(self, img):
        angle_2 = random.randint(-self.max_angle,self.max_angle)
        angle_1 = random.randint(-self.max_angle,self.max_angle)
        angle_3 = random.randint(-self.max_angle,self.max_angle)
        img = ndimage.rotate(img, angle_2, axes=(0, 1), reshape=True, order=0)
        img = ndimage.rotate(img, angle_1, axes=(0, 2), reshape=True, order=0)
        img = ndimage.rotate(img, angle_3, axes=(1, 2), reshape=True, order=0)
        return img

class RandomRotationXY(object):
    def __init__(self,max_angle):
        self.max_angle = max_angle
    def __call__(self, img):
        angle = random.randint(-self.max_angle,self.max_angle)
        img = ndimage.rotate(img, angle, axes=(0, 1), reshape=True, order=0)
        return img

class RandomColorScale3D(object):
    def __init__(self,max_scale,seg):
        self.max_scale = max_scale
        self.seg = seg
    def __call__(self, img):
        x,y,z,c = img.shape
        random_list = []
        for i in range(c-1):
            random_list.append((2*self.max_scale * random.random()-self.max_scale)+1)

        if self.seg:
            random_list.append(1.0)
        else:
            random_list.append((2*self.max_scale * random.random()-self.max_scale)+1)
        scale = np.array(random_list).astype('float32')
        scale = scale[None,None,None,:]
        img = img * scale
        return img

class RandomZoom3D(object):
    def __init__(self,max_scale):
        self.max_scale = max_scale
    def __call__(self, img):
        scale = (2*self.max_scale * random.random()-self.max_scale )+1
        img = ndimage.zoom(img, (scale, scale, scale, 1),order=0)
        return img

class ZoomScale3D(object):
    def __init__(self,scale):
        self.scale = scale
    def __call__(self, img):

        img = ndimage.zoom(img, (self.scale, self.scale, self.scale, 1), order=1)
        return img

class RandomShift3D(object):
    def __init__(self,max_shift_x, max_shift_y, max_shift_z, ):
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        self.max_shift_z = max_shift_z
    def __call__(self, img):
        shift_x = random.randint(-self.max_shift_x, self.max_shift_x)
        shift_y = random.randint(-self.max_shift_y, self.max_shift_y)
        shift_z = random.randint(-self.max_shift_z, self.max_shift_z)
        img = ndimage.shift(img, (shift_x, shift_y, shift_z, 0),order=0)
        return img

#

class RandomNoiseXY(object):
    def __init__(self,sigma,seg):
        self.sigma = sigma
        self.seg=seg
    def __call__(self,img):
        if self.seg:
            # last channel is the seg channel
            img_shape = np.shape(img[:,:,:,:-1])
            sigma = random.random() * self.sigma
            noise = np.random.normal(0,sigma,img_shape).astype('float32')
            noise = np.reshape(noise,img_shape)
            noise = np.concatenate([noise,np.zeros_like(noise[:,:,:,-1:])],axis=-1)
            img = img + noise
            img = np.clip(img,0.0,1.0)
        else:
            sigma = random.random() * self.sigma
            img_shape = np.shape(img)
            noise = np.random.normal(0,sigma,img_shape).astype('float32')
            noise = np.reshape(noise,img_shape)
            img = img + noise
            img = np.clip(img,0.0,1.0)
        return img

class RandomMask3D(object):
    def __init__(self,max_length,num,mask_prob):
        self.m_l = max_length
        self.n = num
        self.p = mask_prob
    def __call__(self,img):
        for i in range(self.n):
            if random.random()>1-self.p:
                continue
            x_r,y_r,z_r,_ = np.shape(img)
            z_ratio = z_r/x_r
            z_m_l=round(self.m_l*z_ratio+0.4)
            l_x = random.randint(0, self.m_l)
            l_y = random.randint(0, self.m_l)
            l_z = random.randint(0, z_m_l)
            l_p_x = random.randint(0, x_r - l_x)
            l_p_y = random.randint(0, y_r - l_y)
            l_p_z = random.randint(0, z_r - l_z)
            img[l_p_x:l_x+l_p_x,
                l_p_y:l_y+l_p_y,
                l_p_z:l_z+l_p_z,
                :]=0
            # print('1',z_m_l)
            # print('2',l_p_x, l_x + l_p_x)
            # print('3',l_p_y, l_y + l_p_y)
            # print('4',l_p_z, l_z + l_p_z)
        return img

class RandomColor2rdScale3D(object):
    def __init__(self,max_scale):
        self.max_scale = max_scale
    def __call__(self, img):
        img=np.clip(img,0.0,1.0)
        scale_a = (self.max_scale-1)*random.random()+1
        if random.random()>0.5:
            scale_a = 1/scale_a
        scale_b = (self.max_scale-1)*random.random()+1
        if random.random()>0.5:
            scale_b = 1/scale_b

        img[:,:,:,0]=np.power(img[:,:,:,0],scale_a)
        img[:,:,:,1]=np.power(img[:,:,:,1],scale_b)

        return img

class RandomFlipping(object):
    def __init__(self,prob):
        self.prob = prob
    def __call__(self, img):
        rand_x = random.random()
        rand_y = random.random()
        rand_z = random.random()

        if rand_x < self.prob:
            img = img[::-1, :, :, :]
        if rand_y < self.prob:
            img = img[:, ::-1, :, :]
        if rand_z < self.prob:
            img = img[:, :, ::-1, :]

        return img

class RandomTransposingXY(object):
    def __init__(self,prob):
        self.prob = prob
    def __call__(self, img):
        rand = random.random()
        if rand < self.prob:
            img = np.transpose(img,[1,0,2,3])
        return img