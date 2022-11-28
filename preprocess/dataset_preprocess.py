import re
import os
import cv2
import pdb
import glob
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def csawcc_preprocess(image_folder, mask_folder, save_folder=None, mask_postfix="mask", train_ratio=0.9):
    """
    Create a metadata file for CSAW_CC dataset
    """

    df = pd.DataFrame(columns=['patient_id', 'date', 'laterality', 'view', 'num', 'data_path','mask_path','mode'], )
    num_of_train_data = int(train_ratio*len(os.listdir(image_folder)))

    for i, file in enumerate(os.listdir(image_folder)):
        d = os.path.join(image_folder, file)
        if os.path.isfile(d):
            data_path = os.path.abspath(d)
            patient_info = file.replace(".png","").split("_")
            mask_path = os.path.join(mask_folder, "_".join(patient_info + [mask_postfix]) + ".png")
            
            if not os.path.isfile(mask_path):
                print("Cannot find mask of {} file!!!".format(file))
                break
            
            mode = "train" if i < num_of_train_data else "test"
            df = pd.concat([pd.DataFrame([patient_info + [data_path, mask_path, mode]], columns=df.columns), df], ignore_index=True)
    

    if save_folder == None:
        csv_file = os.path.join(os.getcwd,"preprocess","metadata.csv")
    else:
        csv_file = os.path.join(save_folder,"metadata.csv")

    df.to_csv(csv_file,index_label="id")
            



def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    
    csawcc_preprocess(
        image_folder=r"E:\csawcc_png16\images_and_masks\images_and_masks\images\png16",
        mask_folder=r"E:\csawcc_png16\images_and_masks\images_and_masks\masks",
        save_folder=r"D:\framework\\breast_cancer_segmentation\preprocess"
    )

    df = pd.read_csv(r'D:\framework\\breast_cancer_segmentation\preprocess\metadata.csv')

    pass

    parser = argparse.ArgumentParser(
        description='Data process for .')
    parser.add_argument('--dataset', type=str, default='none',
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='../none',
                        help='path to the dataset')
    parser.add_argument('--annotation-prefix', type=str, default='annotations/manual/{}.corpus.csv',
                        help='annotation prefix')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()

