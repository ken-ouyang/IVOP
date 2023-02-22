import numpy as np
import argparse
import math

from glob import glob
from ntpath import basename
# from scipy.misc import imread
from matplotlib.pyplot import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray
from skimage.transform import resize
import os
import matplotlib


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
   # parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    #parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)



args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

# path_true = '/data_new/tengfei/generative_inpainting_ms_att_test/results-m/2' #args.data_path
# path_pred = '/data_new/tengfei/generative_inpainting_ms_att_test/results-m/1' #args.output_path
path_true = '/disk1/ouyang/proj/invert_image_op/Invertible-Image-Rescaling/experiments/01_WMF_twoBranches_4_1/val_images' #args.data_path
path_pred = '/disk1/ouyang/proj/invert_image_op/Invertible-Image-Rescaling/experiments/RAISE_gray_twoBranches_4_1/val_images' #args.output_path
# path_true = 'test_result/camel_240p_gt' #args.data_path
# path_pred = 'test_result/camel_240p' #args.output_path

psnr = []
ssim = []
mae = []
names = []
index = 1


psnr2 = []
ssim2 = []
mae2 = []
#files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
files = os.listdir(path_pred)
#pred_list = []
#for root in sorted(os.listdir(path_pred)):
#    #for file in files:
#    f_path = os.path.join(path_pred, root, root + '_40000.jpg') #_forwLR_
#    pred_list.append(f_path)

for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)

    # img_gt = (imread(str(fn), mode='L') / 255.0).astype(np.float32)
    # img_pred = (imread(path_pred + '/' + basename(str(fn)), mode='L') / 255.0).astype(np.float32)
    #img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    #img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)   
    #img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    
    img_gt = (imread(path_pred + '/' + name + '/' + name + '_LR_ref_10000.jpg' ) / 255.0).astype(np.float32)
    img_pred = (imread(path_pred + '/' + name + '/' + name + '_forwLR_20000.jpg' ) / 255.0).astype(np.float32)
    img_gt2 = (imread(path_pred + '/' + name + '/' + name + '_GT_10000.jpg' ) / 255.0).astype(np.float32)
    img_pred2 = (imread(path_pred + '/' + name + '/' + name + '_20000.jpg' ) / 255.0).astype(np.float32)    
    
#    img_gt = (imread(path_pred + '/' + name + '/' + name + '_LR_ref_20000.png' ) )
#    matplotlib.image.imsave("./temp.jpg", img_gt)
#    img_gt = (imread("./temp.jpg" ) / 255.0).astype(np.float32)
#
#    img_pred = (imread(path_pred + '/' + name + '/' + name + '_forwLR_80000.png' ) )
#    matplotlib.image.imsave("./temp.jpg", img_pred)
#    img_pred = (imread("./temp.jpg" ) / 255.0).astype(np.float32)
#        
#    img_gt2 = (imread(path_pred + '/' + name + '/' + name + '_GT_20000.png' ) )
#    matplotlib.image.imsave("./temp.jpg", img_gt2)
#    img_gt2 = (imread("./temp.jpg" ) / 255.0).astype(np.float32)    
#    
#    img_pred2 = (imread(path_pred + '/' + name + '/' + name + '_80000.png' ) )
#    matplotlib.image.imsave("./temp.jpg", img_pred2)
#    img_pred2 = (imread("./temp.jpg" ) / 255.0).astype(np.float32)  
          
    #img_pred = (imread(path_pred + '/' + name + '/' + name + '_forwLR_100000.png' ) / 255.0).astype(np.float32)
    
    #img_pred = resize(img_pred, (img_gt.shape[0], img_gt.shape[1])) 
#    img_gt = (imread(path_pred + '/' + name[:-4] + '/' + name[:-4] + '_GT_10000.jpg' ) / 255.0).astype(np.float32)
#    img_pred = (imread(path_pred + '/' + name[:-4] + '/' + name[:-4] + '_440000.jpg' ) / 255.0).astype(np.float32)    
#    #img_pred = resize(img_pred, (380, 380))
#    #img_gt = resize(img_gt, (380, 380))
    
#for i in range(995):
#    img_pred = (imread(path_pred + '/' + 'a4%03d_output.jpg' % i ) / 255.0).astype(np.float32)
#    img_gt = (imread(path_pred + '/' + 'a4%03d_gt.jpg' % i ) / 255.0).astype(np.float32)
    
    psnr2.append(compare_psnr(img_gt2, img_pred2, data_range=1))
    ssim2.append(compare_ssim(img_gt2, img_pred2, data_range=1, win_size=51, multichannel=True))
    mae2.append(compare_mae(img_gt2, img_pred2))

    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51, multichannel=True))
    mae.append(compare_mae(img_gt, img_pred))
    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
      
            "PSNR2: %.4f" % round(np.mean(psnr2), 4),
            "SSIM2: %.4f" % round(np.mean(ssim2), 4),
            "MAE2: %.4f" % round(np.mean(mae2), 4),
        )
    index += 1

#np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4),
    "PSNR2: %.4f" % round(np.mean(psnr2), 4),
    "SSIM2: %.4f" % round(np.mean(ssim2), 4)
)