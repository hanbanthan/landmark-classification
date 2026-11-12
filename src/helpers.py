from io import BytesIO
import urllib.request
from zipfile import ZipFile
import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

import numpy as np
import random

def setup_env():
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. Will use CPU (slow)")
    
    # seed random generator for repeatability
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    download_and_extract()
    compute_mean_and_std()
    
    os.makedirs("checkpoints", exist_ok=True)
    
def get_data_location():
    if os.path.exists("landmark_images"):
        data_folder = "landmark_images"
    else:
        raise IOError("Please download the dataset first")
    
    return data_folder
    
def download_and_extract(
    url="https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",
):
    try:
        location = get_data_location()
    except IOError:
        print(f"Downloading and unzipping {url}. This will take a while...")
        
        with urllib.request.urlopen(url) as resp:
            
            with ZipFile(BytesIO(resp.read())) as fp:
                
                fp.extractall(".")
                
        print("done")
    
    else:
        
        print(
            "Dataset already downloaded. If you need re-download, "
            f"please delete the directory {location}"
        )
    
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """
    
    cache_file = "mean_and_std.pt"
    
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)
        
        return d["mean"], d["std"]
    
    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=0
    )
    
    mean = 0.0
    
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_examples = images.size(0)
        images = images.view(batch_examples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)
    
    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_examples = images.size(0)
        images = images.view(batch_examples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()
        
    std = torch.sqrt(var / (npix / 3))
    
    torch.save({"mean": mean, "std": std}, cache_file)
    
    return mean, std
    
def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")
    
    if group_name.lower() == "loss":
        ax.set_ylim([None, 4.5])
        

    
    
    
    