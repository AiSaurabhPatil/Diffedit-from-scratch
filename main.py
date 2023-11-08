from masking import create_mask
from diffedit import prompt_to_img_diffedit
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
from fastdownload import FastDownload

def load_image(p):
    '''
    Function to load images from a defined path
    '''
    return Image.open(p).convert('RGB').resize((512,512))

def diffEdit(init_img, rp , qp, g=7.5, seed=100, strength =0.7, steps=70, dim=512):
    
    ## Step 1: Create mask
    mask = create_mask(init_img=init_img, refer_prompt=rp, query_prompt=qp)
    
    ## Step 2 and 3: Diffusion process using mask
    output = prompt_to_img_diffedit(
        refer_prompt=rp, 
        query_prompt=qp, 
        init_img=init_img, 
        mask = mask, 
        g=g, 
        seed=seed,
        strength =strength, 
        steps=steps, 
        dim=dim)
    return output


def plot_diffEdit(init_img, output):
    ## Plotting side by side
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    
    ## Visualizing initial image
    axs[0].imshow(init_img)
    axs[0].set_title(f"Initial image")
    
    ## Visualizing initial image
    axs[1].imshow(output[0])
    axs[1].set_title(f"DiffEdit output")
    

p = FastDownload().download('https://raw.githubusercontent.com/johnrobinsn/diffusion_experiments/main/images/bowloberries_scaled.jpg')
init_img = load_image(p)
output = diffEdit(
  init_img, 
  rp = ['Bowl of Strawberries'], 
  qp=['Bowl of Grapes']
)
plot_diffEdit(init_img, output)