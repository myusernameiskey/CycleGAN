import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, target_transform, target_transform_resize
from data.image_folder import make_dataset_cardiac
from PIL import Image
import PIL
import random
import numpy as np
import scipy.io as sio
import torch

import hdf5storage
from os import listdir
from os.path import join
import sys

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


def load_mat(filepath, normal_Option):
    #img = loadmat(filepath) # path of .mat files to be loaded
    img = hdf5storage.loadmat(filepath)
    matname = list(img.keys())[0] # take variable name (defined in Matlab)
    img = np.array(img[matname])
    img = img[:,:,np.newaxis]
    img = img.astype("float32")
    ###################### do normalization here ##############################
    
    if normal_Option:
        std = np.std(img)
        mean = np.mean(img)
        img = (img-mean)/std    
    
  #  img = (img+1000)/(7000)
     
    ###################### do normalization here ##############################    
    ToPIL = transforms.ToPILImage(mode = 'F')
 
    img = ToPIL(img)
    return img

def load_image_train(seed, path, target_transform, normal_Option, index):
    random.seed(seed)        
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    return target_transform(load_mat(path[index], normal_Option))

def import_3C(index, image_IN_filenames, image_OUT_filenames, normal_Option, seed, target_transform):
    slice_number = ''.join([num for num in image_IN_filenames[index] if num.isdigit()])
    if '0001' in slice_number:
        index = index+1
    elif '0800' in slice_number:
        index = index-1

    A_top_image = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index-1)
    A_middle_image = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index)
    A_bottom_image = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index+1)

    B_top_image = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index-1)
    B_middle_image = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index)
    B_bottom_image = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index+1)

    return A_top_image, A_middle_image, A_bottom_image, B_top_image, B_middle_image, B_bottom_image

def import_5C(index, image_IN_filenames, image_OUT_filenames, normal_Option, seed, target_transform):
    slice_number = ''.join([num for num in image_IN_filenames[index] if num.isdigit()])
    if '0001' in slice_number:
        index = index+2
    elif '0002' in slice_number:
        index = index+1
    elif '0799' in slice_number:
        index = index-1
    elif '0800' in slice_number:
        index = index-2

    A_top_image2 = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index-2)
    A_top_image = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index-1)
    A_middle_image = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index)
    A_bottom_image = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index+1)
    A_bottom_image2 = load_image_train(seed, image_IN_filenames, target_transform, normal_Option, index+2)

    B_top_image2 = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index-2)
    B_top_image = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index-1)
    B_middle_image = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index)
    B_bottom_image = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index+1)
    B_bottom_image2 = load_image_train(seed, image_OUT_filenames, target_transform, normal_Option, index+2)

    return A_top_image2, A_top_image, A_middle_image, A_bottom_image, A_bottom_image2, B_top_image2, B_top_image, B_middle_image, B_bottom_image, B_bottom_image2

def import_3C_test(index, image_IN_filenames, image_OUT_filenames, normal_Option):
    slice_number = ''.join([num for num in image_IN_filenames[index] if num.isdigit()])
    if '0001' in slice_number:
        A_top_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index+1], normal_Option))
    elif '0800' in slice_number:
        A_top_image = (load_mat(image_IN_filenames[index-1], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index], normal_Option))
    else:
        A_top_image = (load_mat(image_IN_filenames[index-1], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index+1], normal_Option))
    B_middle_image = (load_mat(image_OUT_filenames[index], normal_Option))

    return A_top_image, A_middle_image, A_bottom_image, B_middle_image

def import_5C_test(index, image_IN_filenames, image_OUT_filenames, normal_Option):
    slice_number = ''.join([num for num in image_IN_filenames[index] if num.isdigit()])
    if '0001' in slice_number:
        A_top_image2 = (load_mat(image_IN_filenames[index], normal_Option))
        A_top_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index+1], normal_Option))
        A_bottom_image2 = (load_mat(image_IN_filenames[index+2], normal_Option)) 
    elif '0002' in slice_number:
        A_top_image2 = (load_mat(image_IN_filenames[index-1], normal_Option))
        A_top_image = (load_mat(image_IN_filenames[index-1], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index+1], normal_Option))
        A_bottom_image2 = (load_mat(image_IN_filenames[index+2], normal_Option)) 
    elif '0799' in slice_number:
        A_top_image2 = (load_mat(image_IN_filenames[index-2], normal_Option))
        A_top_image = (load_mat(image_IN_filenames[index-1], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index+1], normal_Option))
        A_bottom_image2 = (load_mat(image_IN_filenames[index+1], normal_Option))
                           
    elif '0800' in slice_number:
        A_top_image2 = (load_mat(image_IN_filenames[index-2], normal_Option))
        A_top_image = (load_mat(image_IN_filenames[index-1], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image2 = (load_mat(image_IN_filenames[index], normal_Option)) 

    else:
        A_top_image2 = (load_mat(image_IN_filenames[index-2], normal_Option))
        A_top_image = (load_mat(image_IN_filenames[index-1], normal_Option))
        A_middle_image = (load_mat(image_IN_filenames[index], normal_Option))
        A_bottom_image = (load_mat(image_IN_filenames[index+1], normal_Option))
        A_bottom_image2 = (load_mat(image_IN_filenames[index+2], normal_Option)) 
    B_middle_image = (load_mat(image_OUT_filenames[index], normal_Option))

    return A_top_image2, A_top_image, A_middle_image, A_bottom_image, A_bottom_image2, B_middle_image

class Dental_LD_25D(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if opt.phase == 'Training':
            self.dir_A = os.path.join(opt.dataroot, opt.phase , 'in')
            self.dir_B = os.path.join(opt.dataroot, opt.phase , 'target')
        elif opt.phase == 'Test':
            self.dir_A = os.path.join(opt.dataroot, opt.phase , 'in', opt.testsetname_in)
            self.dir_B = os.path.join(opt.dataroot, opt.phase , 'target', opt.testsetname_tar)
            
        #self.transform = target_transform()
        if opt.resize_or_crop == 'resize_and_crop':
            self.target_transform = target_transform_resize(opt)
        else:
            self.target_transform = target_transform(opt)

        self.crop_size = opt.fineSize
        self.normal_Option = opt.Normal_Option
        self.residual_L= opt.Residual_L
        self.remove_back_patch = opt.Remove_back_patch
               
        self.image_IN_filenames = [join(self.dir_A, x) for x in listdir(self.dir_A) if is_image_file(x)] # load files from filelist.
        self.image_OUT_filenames = [join(self.dir_B, x) for x in listdir(self.dir_B) if is_image_file(x)] # load files from filelist.
        
        self.Input_nc = opt.inoutput_nc_25D

        
    def __getitem__(self, index):
        if (self.remove_back_patch) and (self.opt.phase == 'Training'):
            count = 0
            while count <1:
                seed = random.randint(0**16)
                if self.Input_nc == 3:
                    A_top_image, A_middle_image, A_bottom_image, B_top_image, B_middle_image, B_bottom_image = import_3C(index, self.image_IN_filenames, self.image_OUT_filenames, self.normal_Option, seed, self.target_transform)
                elif self.Input_nc == 5:
                    A_top_image2, A_top_image, A_middle_image, A_bottom_image, A_bottom_image2, B_top_image2, B_top_image, B_middle_image, B_bottom_image, B_bottom_image2 = import_5C(index, self.image_IN_filenames, self.image_OUT_filenames, self.normal_Option, seed, self.target_transform)
                else:
                    print("Check input channel size")
                    sys.exit() 
            
                if np.logical_or((sum(sum(np.array(A_middle_image)<-800)) > (self.crop_size**2)/2.5 ),(sum(sum(np.array(B_middle_image)<-800)) > (self.crop_size**2)/2.5 )) :
                    continue
                else: 
                    count += 1  

            t = transforms.ToTensor()
            if self.residual_L:
                target = t(B_middle_image)-t(A_middle_image)
            else:
                target = t(B_middle_image)

            if self.Input_nc == 3:
                return {'A': t(A_middle_image) , 'B': target, 
                    'A_25D': torch.cat([t(A_top_image), t(A_middle_image),t(A_bottom_image)],dim=0),
                    'B_25D': torch.cat([t(B_top_image), t(B_middle_image),t(B_bottom_image)],dim=0),
                    'A_paths': self.image_IN_filenames[index], 'B_paths': self.image_OUT_filenames[index], 'A_max': A_max}
            elif self.Input_nc == 5:
                return {'A': t(A_middle_image) , 'B': target, 
                    'A_25D': torch.cat([t(A_top_image2), t(A_top_image), t(A_middle_image),t(A_bottom_image), t(A_bottom_image2)],dim=0),
                    'B_25D': torch.cat([t(B_top_image2), t(B_top_image), t(B_middle_image),t(B_bottom_image), t(B_bottom_image2)],dim=0),
                    'A_paths': self.image_IN_filenames[index], 'B_paths': self.image_OUT_filenames[index], 'A_max': A_max}
        else:
            if self.opt.phase == 'Training':

                seed = random.randint(0,2**16)
                if self.Input_nc == 3:
                    A_top_image, A_middle_image, A_bottom_image, B_top_image, B_middle_image, B_bottom_image = import_3C(index, self.image_IN_filenames, self.image_OUT_filenames, self.normal_Option, seed, self.target_transform)
                elif self.Input_nc == 5:
                    A_top_image2, A_top_image, A_middle_image, A_bottom_image, A_bottom_image2, B_top_image2, B_top_image, B_middle_image, B_bottom_image, B_bottom_image2 = import_5C(index, self.image_IN_filenames, self.image_OUT_filenames, self.normal_Option, seed, self.target_transform)
                else:
                    print("Check input channel size")
                    sys.exit() 

                A_max = float(np.amax(A_middle_image))    

                t = transforms.ToTensor()
                if self.residual_L:
                    target = t(B_middle_image)-t(A_middle_image)
                else:
                    target = t(B_middle_image)
                        
                if self.Input_nc == 3:
                    return {'A': t(A_middle_image) , 'B': target, 
                        'A_25D': torch.cat([t(A_top_image), t(A_middle_image),t(A_bottom_image)],dim=0),
                        'B_25D': torch.cat([t(B_top_image), t(B_middle_image),t(B_bottom_image)],dim=0),
                        'A_paths': self.image_IN_filenames[index], 'B_paths': self.image_OUT_filenames[index], 'A_max': A_max}
                elif self.Input_nc == 5:
                    return {'A': t(A_middle_image) , 'B': target, 
                        'A_25D': torch.cat([t(A_top_image2), t(A_top_image), t(A_middle_image), t(A_bottom_image), t(A_bottom_image2)],dim=0),
                        'B_25D': torch.cat([t(B_top_image2), t(B_top_image), t(B_middle_image), t(B_bottom_image), t(B_bottom_image2)],dim=0),
                        'A_paths': self.image_IN_filenames[index], 'B_paths': self.image_OUT_filenames[index], 'A_max': A_max}
                # 
                
            elif self.opt.phase == 'Test':
                if self.Input_nc == 3:
                    A_top_image, A_middle_image, A_bottom_image, B_middle_image = import_3C_test(index, self.image_IN_filenames, self.image_OUT_filenames, self.normal_Option)

                elif self.Input_nc == 5:
                    A_top_image2, A_top_image, A_middle_image, A_bottom_image, A_bottom_image2, B_middle_image = import_5C_test(index, self.image_IN_filenames, self.image_OUT_filenames, self.normal_Option)
                else:
                    print("Check input channel size")
                    sys.exit() 
                A_max = float(np.amax(A_middle_image))   
                t = transforms.ToTensor()
                if self.residual_L:
                    target = t(B_middle_image)-t(A_middle_image)
                else:
                    target = t(B_middle_image)

                
                if self.Input_nc == 3:
                    return {'A': t(A_middle_image) , 'B': target, 
                        'A_25D': torch.cat([t(A_top_image), t(A_middle_image),t(A_bottom_image)],dim=0),
                        'A_paths': self.image_IN_filenames[index], 'B_paths': self.image_OUT_filenames[index], 'A_max': A_max}
                elif self.Input_nc == 5:
                    return {'A': t(A_middle_image) , 'B': target, 
                        'A_25D': torch.cat([t(A_top_image2), t(A_top_image), t(A_middle_image),t(A_bottom_image), t(A_bottom_image2)],dim=0),
                        'A_paths': self.image_IN_filenames[index], 'B_paths': self.image_OUT_filenames[index], 'A_max': A_max}

    def __len__(self):
        return len(self.image_IN_filenames)
    
    def name(self):
        return 'DentalDataset'




    
    
    