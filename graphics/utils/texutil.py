
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

fix_gaussian_filter = transforms.GaussianBlur(5, sigma=1)

def apply_filter(filter, img):
    img = img.permute(0, 3, 1, 2)
    img = filter(img)
    return img.permute(0, 2, 3, 1).contiguous()

def create_mipmap(texs, max_mip_level=4, filter=None):
    if filter is None:
        filter = fix_gaussian_filter

    # Create a Mipmap pyramid with 4 levels
    mipmap_levels = [texs]
    for i in range(max_mip_level):
        b, h, w, c = mipmap_levels[-1].shape
        res = h
        # Apply Gaussian blur for downsampling
        filtered = apply_filter(
            transforms.Compose([filter, transforms.Resize(res//2)]),
            mipmap_levels[-1]
        )
        mipmap_levels.append(filtered)
    
    mipmap_levels.pop(0)
    return mipmap_levels