from fastdownload import FastDownload
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from masking import create_mask

def load_image(p):
    '''
    Function to load images from a defined path
    '''
    return Image.open(p).convert('RGB').resize((512,512))


p = FastDownload().download('https://images.pexels.com/photos/1996333/pexels-photo-1996333.jpeg?cs=srgb&dl=pexels-helena-lopes-1996333.jpg&fm=jpg&_gl=1*1pc0nw8*_ga*OTk4MTI0MzE4LjE2NjY1NDQwMjE.*_ga_8JE65Q40S6*MTY2Njc1MjIwMC4yLjEuMTY2Njc1MjIwMS4wLjAuMA..')
init_img = load_image(p)
init_img

mask = create_mask(init_img=init_img, refer_prompt=["a horse image"], query_prompt=["a zebra image"], n=10)


plt.imshow(np.array(init_img), cmap='gray') # I would add interpolation='none'
plt.imshow(
    Image.fromarray(mask).resize((512,512)), ## Scaling the mask to original size
    cmap='cividis', 
    alpha=0.5*(np.array(Image.fromarray(mask*255).resize((512,512))) > 0)  
)