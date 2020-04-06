import os
import cv2
from utils.disp_func import img_hml_plt
from utils.base_func import altmk_outdirs, prog_print

"""This SCRIPT automatically create images in defined directory with labels plotted on.
"""
""" Parameters:
        img_path: {STR} Folder with image files.
        hml_path: {STR} Folder with corresponding label files, use the same as [img_path] if left blank.
        prefix: {STR} Prefix of target files.
        extension: {STR} File extension of target image files.
        label_name: {STR or LIST of STR or None} The name of labels to be plotted, plot all if set to None.
        color_list: {DICT} Dictionary of colors linked with each label, use COLORMAP_JET if set to None.
        out_path: {STR} Folder where all labelled images will be stored, use [img_path + "/lbl_img"] if left blank.
"""

# Parameters input  -------------------------------------------------------------------------------------------------- #
img_path = "./img_data/"
hml_path = ""
prefix = ""
extension = ".png"
label_name = None
color_list = None
out_path = ""
# -------------------------------------------------------------------------------------------------------------------- #


# Standardize file input path
if not img_path.endswith("/"):
    img_path += "/"
if (hml_path == str()) or (hml_path is None):
    hml_path = img_path
elif not hml_path.endswith("/"):
    hml_path += "/"

# Set labelled images output directory
out_path = altmk_outdirs(out_path, img_path + "lbl_img/", err_msg="Invalid output path, function out!")

# Get all file-pair names, images as reference
names = [os.path.splitext(f)[0] for f in os.listdir(img_path) if f.startswith(prefix) and f.endswith(extension)]

# Process loop
tot = len(names)
print("Creating %d labelled images." % tot)
i = 1    # INIT VAR
for n in names:
    # Join to full file path
    img_file = img_path + n + ".png"
    hml_file = hml_path + n + ".pkl"
    # Plot and save labels to image
    img, title = img_hml_plt(img_file, hml_file, label_name, color_list)
    out_file = os.path.join(out_path, title) + "_labelled.png"
    cv2.imwrite(out_file, img)
    # Print progress
    prog_print(i, tot, "Progress:", "finished.  ")
    i += 1
print("Process DONE!")
