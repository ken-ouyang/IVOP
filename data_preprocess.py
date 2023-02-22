import cv2
import numpy as np
import pathlib
import glob

def read_img(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return img 

image_path = "night2day/train"
image_list = sorted(glob.glob("%s/*"%image_path))

# new_folder = "night2day/test_night"
new_folder = "night2day/train_day"
pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)

for annt_path in image_list:
    new_path = "%s/%s" % (new_folder, annt_path.split("/")[-1])
    image = read_img(annt_path)
    # new_image = image[:,:256,:]
    new_image = image[:,256:,:]
    cv2.imwrite(new_path, new_image)
