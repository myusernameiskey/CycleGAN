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

        # max = np.max(img)
        # min = np.min(img)
        # img = (img-min)/(max-min) 
        # img = (img-0.5)*2
    
  #  img = (img+1000)/(7000)
     
    ###################### do normalization here ##############################    
    ToPIL = transforms.ToPILImage(mode = 'F')
#        
    img = ToPIL(img)
    return img



class Dental_LD(BaseDataset):
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


        
    def __getitem__(self, index):
        if self.remove_back_patch:
            count = 0
            while count <1:
                seed = random.randint(0,2**16)
                
                random.seed(seed)        
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                in_image = self.target_transform(load_mat(self.image_IN_filenames[index], self.normal_Option))
                
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                out_image = self.target_transform(load_mat(self.image_OUT_filenames[index], self.normal_Option))
                
                if np.logical_or((sum(sum(np.array(in_image)<-800)) > (self.crop_size**2)/2.5 ),(sum(sum(np.array(out_image)<-800)) > (self.crop_size**2)/2.5 )) :
                    continue
                else: 
                    count += 1  

            t = transforms.ToTensor()
            if self.residual_L:
                target = t(out_image)-t(in_image)
            else:
                target = t(out_image)
            
            return t(in_image), target
        else:
            
            seed = random.randint(0,2**16)
                
            random.seed(seed)        
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if self.opt.phase == 'Training':
                in_image = self.target_transform(load_mat(self.image_IN_filenames[index], self.normal_Option))
            elif self.opt.phase == 'Test':
                in_image = (load_mat(self.image_IN_filenames[index], self.normal_Option))
                
            A_max = float(np.amax(in_image))    
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            if self.opt.phase == 'Training':
                out_image = self.target_transform(load_mat(self.image_OUT_filenames[index], self.normal_Option))
            elif self.opt.phase == 'Test':
                out_image = (load_mat(self.image_OUT_filenames[index], self.normal_Option))
                
                
            t = transforms.ToTensor()
            if self.residual_L:
                target = t(out_image)-t(in_image)
            else:
                target = t(out_image)
            
            #return t(in_image), target
            return {'A': t(in_image), 'B': target,
                    'A_paths': self.image_IN_filenames[index], 'B_paths': self.image_OUT_filenames[index], 'A_max': A_max}

    def __len__(self):
        return len(self.image_IN_filenames)
    
    def name(self):
        return 'DentalDataset'




    
    
    