import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html2
import numpy as np
import ipdb
import torch
import math
from multiprocessing import freeze_support

        

def main(): 
    freeze_support()
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
     
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html2.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    exp_dir = web_dir+"/recon_mat"
    
    opt.how_many = len(dataset)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    save_i = 1;
    # test
    for i, data in enumerate(dataset):
        
        #if i >= opt.how_many:
        #    break
    
        patient_case = data.get('A_paths')[0]
        tmp = patient_case.split('\\')
        exp_dir_ = exp_dir + "/" + tmp[-2]
    
        if not os.path.exists(os.path.join(exp_dir_,'Input')) |os.path.exists(os.path.join(exp_dir_,'Target'))|os.path.exists(os.path.join(exp_dir_,'Output')):
            save_i = 0;
            os.makedirs(os.path.join(exp_dir_,'Input'))
            os.makedirs(os.path.join(exp_dir_,'Target'))
            os.makedirs(os.path.join(exp_dir_,'Output'))

        if opt.model != 'cycle_gan_25D':
            model.set_input_test(data['A'])
        else:
            model.set_input_test(data)

        model.test()
        visuals = model.get_current_visuals_test()
        
        
        recon_img = np.array(visuals['fake_B'].cpu())
        #recon_img = (recon_img/2+0.5)*data['A_max']
        _, _, h, w = np.shape(recon_img)
        recon_img = np.reshape(recon_img, (h,w))

        
        
        visualizer.save_result_mat_for_Testing(data['A'].squeeze(), 'Input', exp_dir_, '{0:03d}'.format(save_i))
        visualizer.save_result_mat_for_Testing(data['B'].squeeze(), 'Target', exp_dir_, '{0:03d}'.format(save_i))
        visualizer.save_result_mat_for_Testing(recon_img, 'Output', exp_dir_, '{0:03d}'.format(save_i))
        save_i = save_i + 1;
    
        if save_i % 50 == 0:
            print(tmp[-3] + ': ' + str(save_i) + ' slices recon')
    
    webpage.save()

    

if __name__ == "__main__":
    main()