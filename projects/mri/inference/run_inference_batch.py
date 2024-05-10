"""
Run MRI inference data in the batch mode
"""
import os
import argparse
import copy
from time import time

import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

import nibabel as nib

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

MRI_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(MRI_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[3].resolve()
sys.path.append(str(REPO_DIR))

from utils import *
from inference import apply_model, load_model, apply_model_3D

# -------------------------------------------------------------------------------------------------
# setup for testing from cmd

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI test evaluation")

    parser.add_argument("--input_dir", default=None, help="folder to load the batch data, go to all subfolders")
    parser.add_argument("--output_dir", default=None, help="folder to save the data; subfolders are created for each case")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="scaling factor to adjust model strength; higher scaling means lower strength")
    parser.add_argument("--im_scaling", type=float, default=1.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=1.0, help="extra scaling applied to gmap")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--batch_size", type=int, default=16, help='after loading a batch, start processing')
    parser.add_argument("--overlap", nargs='+', type=int, default=None, help='overlap for (T, H, W), e.g. (2, 8, 8), (0, 0, 0) means no overlap')

    parser.add_argument("--num_batches_to_process", type=int, default=-1, help='number of batches to be processed')

    parser.add_argument("--input_fname", type=str, default="im", help='input file name')
    parser.add_argument("--gmap_fname", type=str, default="gfactor", help='gmap input file name')

    parser.add_argument("--model_type", type=str, default=None, help="if set, overwrite the config setting, STCNNT_MRI or MRI_hrnet, MRI_double_net")

    return parser.parse_args()

def check_args(args):
    """
    checks the cmd args to make sure they are correct
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - args (Namespace): the checked and updated argparse for MRI
    """
    assert args.saved_model_path.endswith(".pt") or args.saved_model_path.endswith(".pts") or args.saved_model_path.endswith(".onnx") or args.saved_model_path.endswith(".pth"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\" or \"*.onnx\" or \"*.pth\""

    # get the args path
    fname = os.path.splitext(args.saved_model_path)[0]
    args.saved_model_config  = fname + '.yaml'

    return args

# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def process_a_batch(args, model, config, images, selected_cases, gmaps, device):

    if args.overlap:
        overlap_used = tuple(args.overlap)
    else:
        overlap_used = None

    for ind in range(len(images)):
        case_dir = selected_cases[ind]
        print(f"-----------> Process {selected_cases[ind]} <-----------")

        image = images[ind]
        gmap = gmaps[ind]

        if len(image.shape) == 3 and gmap.ndim==3 and gmap.shape[2]==image.shape[2]:
            output = apply_model_3D(image, model, gmap, config=config, scaling_factor=args.scaling_factor, device=get_device(), overlap=overlap_used, verbose=True)
            print(f"3D mode, {args.input_dir}, images - {image.shape}, gmap - {gmap.shape}, median gmap {np.median(gmap)}")
        else:
            output = apply_model(image, model, gmap, config=config, scaling_factor=args.scaling_factor, device=device, overlap=overlap_used, verbose=True)

        case = os.path.basename(case_dir)
        output_dir = os.path.join(args.output_dir, case)
        os.makedirs(output_dir, exist_ok=True)

        save_inference_results(image, output, gmap, output_dir)

        print("--" * 30)

def main():

    args = check_args(arg_parser())
    print(args)

    print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
    model, config = load_model(args.saved_model_path)

    config.height = config.mri_height
    config.width = config.mri_width

    patch_size_inference = args.patch_size_inference

    config.pad_time = args.pad_time
    config.ddp = False

    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference

    device=get_device()

    os.makedirs(args.output_dir, exist_ok=True)

    # load the cases
    case_dirs = fast_scandir(args.input_dir)
    print(case_dirs)

    selected_cases = []
    images = []
    gmaps = []

    num_batches_processed = 0
    
    with tqdm(total=len(case_dirs), bar_format=get_bar_format()) as pbar:
        for c in case_dirs:
            fname = os.path.join(c, f"{args.input_fname}_real.npy")
            if os.path.isfile(fname):
                image = np.load(os.path.join(c, f"{args.input_fname}_real.npy")) + np.load(os.path.join(c, f"{args.input_fname}_imag.npy")) * 1j
                image /= args.im_scaling

                gmap = np.load(f"{c}/{args.gmap_fname}.npy")
                gmap /= args.gmap_scaling

                if len(image.shape) == 3 and gmap.ndim==3 and gmap.shape[2]==image.shape[2]:
                    images.append(image)
                    gmaps.append(gmap)
                    selected_cases.append(c)
                else:
                    if len(image.shape) == 2:
                        image = image[:,:,np.newaxis,np.newaxis]
                    elif len(image.shape) == 3:
                        image = image[:,:,:,np.newaxis]

                    if(image.shape[3]>20):
                        image = np.transpose(image, (0, 1, 3, 2))

                    RO, E1, frames, slices = image.shape
                    print(f"{c}, images - {image.shape}")

                    if(gmap.ndim==2):
                        gmap = np.expand_dims(gmap, axis=2)

                    if gmap.shape[2] != slices:
                        continue
                    else:
                        images.append(image)
                        gmaps.append(gmap)
                        selected_cases.append(c)

            if len(images)>0 and len(images)%args.batch_size==0:
                process_a_batch(args, model, config, images, selected_cases, gmaps, device)
                selected_cases = []
                images = []
                gmaps = []
                
                num_batches_processed += 1
                if args.num_batches_to_process> 0 and num_batches_processed > args.num_batches_to_process:
                    break

            pbar.update(1)

    # process left over cases
    if args.num_batches_to_process <= 0:
        if len(images) > 0:
            process_a_batch(args, model, config, images, selected_cases, gmaps, device)

if __name__=="__main__":
    main()
