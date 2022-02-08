import os
import cv2
import numpy as np
import zlib
import pickle as pkl

def hml_read(hm_lbl_file):
    """ Import a PICKLE HeatMap label file to a Python list of dictionary.
    Args:
        hm_lbl_file (str): Labelling file contained with HeatMap labels.
    Returns:
        list[dict]: List of dictionary with HeatMap label info.
    """
    with open(hm_lbl_file, 'rb') as infile:
        comp = pkl.load(infile)
    hml_data = pkl.loads(zlib.decompress(comp))
    return hml_data

if __name__ == "__main__": 
    file_name = "/home/mshooter/OptiFlex/fluffy_dogs/20220208_video-000-clean_plate-dog2/lbls/20220208_video-000-clean_plate-dog2_frm_34.pkl"
    rgb_name = "/home/mshooter/OptiFlex/fluffy_dogs/20220208_video-000-clean_plate-dog2/imgs/20220208_video-000-clean_plate-dog2_frm_34.png"
    export_data = "../export/"
    hml_data = hml_read(file_name)
    img = cv2.imread(rgb_name)
    for idx, h in enumerate(hml_data): 
        z0,z1 = np.unravel_index(h["heatmap"].argmax(), h["heatmap"].shape)
        img = cv2.circle(img, (z1,z0), radius=2, color=(0,0,255), thickness=2)
    img = cv2.imwrite(os.path.join(export_data,"final.png"),img) 
