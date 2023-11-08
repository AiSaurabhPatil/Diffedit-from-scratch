import torch
from utils import text_encoder , scheduler ,pil_to_latents ,unet
from tqdm.auto import tqdm
import numpy as np
from diffedit import prompt_2_img_i2i


def create_mask(init_img, rp, qp, n=10, s=0.5):
    ## Initialize a dictionary to save n iterations
    diff = {}
    
    ## Repeating the difference process n times
    for idx in range(n):
        ## Creating denoised sample using reference / original text
        orig_noise = prompt_2_img_i2i(prompts=rp, init_img=init_img, strength=s, seed = 100*idx)[0]
        ## Creating denoised sample using query / target text
        query_noise = prompt_2_img_i2i(prompts=qp, init_img=init_img, strength=s, seed = 100*idx)[0]
        ## Taking the difference 
        diff[idx] = (np.array(orig_noise)-np.array(query_noise))
    
    ## Creating a mask placeholder
    mask = np.zeros_like(diff[0])
    
    ## Taking an average of 10 iterations
    for idx in range(n):
        ## Note np.abs is a key step
        mask += np.abs(diff[idx])  
        
    ## Averaging multiple channels 
    mask = mask.mean(0)
    
    ## Normalizing 
    mask = (mask - mask.mean()) / np.std(mask)
    
    ## Binarizing and returning the mask object
    return (mask > 0).astype("uint8")




