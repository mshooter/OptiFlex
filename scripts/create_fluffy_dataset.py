"""
Moira Shooter
Last edited 08/02/2022
Creates for each video the folders, and within those folders there are 2 subfolders
- imgs: where all the rgb images are (symlink) 
- lbls: where all the pkl images are 
"""
import os
import time
import json
import argparse 
import multiprocessing

import cv2
import numpy as np
import copy 
import zlib 
import pickle as pkl
from scipy import stats

def hml_write(dst_hml_file, hml_data): 
    """ Write HeatMap label data to a PICKLE label file.
    Args:
        dst_hml_file (str): Labelling file to write HeatMap labels (*.pkl).
        hml_data (list[dict]): List of dictionary with HeatMap label info.
    Returns:
        bool: File creation status.
    """
    if len(hml_data) != 0:
        comp = zlib.compress(pkl.dumps(hml_data, protocol=2))
        with open(dst_hml_file, 'wb') as outfile:
            pkl.dump(comp, outfile, protocol=2)
        return True
    else:
        print('Empty label data input, file not created!')
        return False

def get_jsonfiles(video_dir, datasetType="clean_plate"): 
    """
    Args: 
        video_dir (str): path to dir with .json files per video 
        datasetType (str): type of dataset (e.g. clean_plate)
    Returns a list of file names based on dataset type
    """
    fnames = []
    for f in sorted(os.listdir(video_dir)): 
        f_split = f.split("-")
        dataset = f_split[2]
        if f.endswith(".json") and dataset==datasetType: 
            fnames.append(f)
    return fnames

def convert_json2hmp(img, kpts, label_names, peak=20.0):
    """
    Args: 
        img (cv2)
        kpts (list): list of keypoints
        peak (int or float): peak value of generated heatmap
    convert jsons into hmps (.pkl)
    """
    # get plot region
    lim_y, lim_x = img.shape[:2]
    x = np.arange(start=0, stop=lim_x, step=1, dtype=np.uint32)
    y = np.arange(start=0, stop=lim_y, step=1, dtype=np.uint32)
    
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    # compute JSON label data to heatmap label data
    hml_data = [] 
    lbl_tmp = {'label': None, 'heatmap': None}
    kpt_data = kpts["keypoints"]
    for k in range(len(kpt_data)): 
        center_x = kpt_data[k]["x"] 
        center_y = kpt_data[k]["y"] 
        mean = [center_x, center_y]
        cov = [[100, 0], [0,100]]
        pdf = stats.multivariate_normal.pdf(xxyy, mean=mean, cov=cov) 
        heatmap = pdf.reshape((lim_x, lim_y))
        trim_lim = 0.1 * pdf.max()
        heatmap = np.where(heatmap < trim_lim, 0, heatmap) 
        hm_peak = peak / np.max(heatmap, axis=None)
        heatmap = np.multiply(heatmap, hm_peak)
        lbl_tmp["label"] = label_names[kpt_data[k]["index"]] 
        lbl_tmp["heatmap"] = heatmap 
        hml_data.append(copy.deepcopy(lbl_tmp))
    return hml_data

parser = argparse.ArgumentParser() 
parser.add_argument("--dataset_dir", help="where the original dataset (perception) is located")
parser.add_argument("--video_dir", help="where all the individual .json per video are located")
parser.add_argument("--out_dir", help="where all the .pkl and rgb files need to be stored")
args = parser.parse_args()


if __name__ == "__main__": 
    
    dataset_dir = "/media/mshooter/TOSHIBA EXT/WhatTheFluff"
    video_dir = "/media/mshooter/TOSHIBA EXT/PhD/IJCV/wtf_annotations"
    out_dir = "/home/mshooter/OptiFlex/fluffy_dogs"
    fnames = get_jsonfiles(video_dir) # list of file names based on datasettype 
    
    # create the directories if does not exist
    for f in fnames: 
        folder = "20220208_{}".format(f.split(".")[0])
        save_dir = os.path.join(out_dir, folder)
        if not os.path.isdir(save_dir): 
            os.makedirs(save_dir)

        # we need to open the json file to require the rgb file and keypoints 
        file_path = os.path.join(video_dir, f)
        f_json = open(file_path) 
        json_data = json.load(f_json)

        # get all the json data
        imgs = json_data["images"] 
        anns = json_data["annotations"]
        templates = json_data["categories"][0]["templates"][0]["key_points"]
        label_names = []
        for t in templates: 
            label_names.append(t["label"])

        # symlink the data and create pkl files
        for idx in range(len(imgs)):
            dogType = folder.split("-")[-1]
            dst_filename = "20220208_{}".format(imgs[idx]["file_name"].split("/")[-1].replace("rgb", os.path.splitext(f)[0]).replace(dogType, "{}_frm".format(dogType))) # dst
            dst_folder_imgs = os.path.join(save_dir, "imgs") 
            dst_folder_lbls = os.path.join(save_dir, "lbls") 

            if not os.path.isdir(dst_folder_imgs):
                os.makedirs(dst_folder_imgs)

            if not os.path.isdir(dst_folder_lbls):
                os.makedirs(dst_folder_lbls)

            dst = os.path.join(dst_folder_imgs, dst_filename)
            src = os.path.join(dataset_dir, imgs[idx]["file_name"]) 
            if not os.path.isfile(dst):
                os.symlink(src, dst)
            
            # get the annotations create .pkl file 
            img = cv2.imread(src)
            hmp_data = convert_json2hmp(img, anns[idx], label_names) 
            hml_write(os.path.join(dst_folder_lbls, dst_filename.replace(".png", ".pkl")), hmp_data)
        f_json.close()
