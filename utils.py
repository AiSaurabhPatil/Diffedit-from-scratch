import torch 
from torchvision import transforms as tfms 
from fastdownload import FastDownload
import numpy as np  
import os
import logging
logging.disable(logging.WARNING)  

from PIL import Image

# Import the CLIP artifacts 
from transformers import CLIPTextModel , CLIPTokenizer
from diffusers import AutoencoderKL , UNet2DConditionModel , DDIMScheduler


def load_artifacts():
    '''
    A function to load all the diffusion artifacts
    '''

    # variation autoencoder to convert image to letent and back to image 
    Vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
                                        subfolder ="vae" ,
                                        torch_dtype = torch.float16).to("cuda") 
    
    # Unet to perform diffusion process 
    UNet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
                                            subfolder ="unet",
                                            torch_dtype = torch.float16).to("cuda") 

    # tokenizer to convert text input into tokens
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14',
                                              torch_dtype = torch.float16)

    # textencoder to convert tokens into vector embeddings 
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                                torch_dtype = torch.float16).to("cuda") 
    
    # scheduler to tune the proportion of noise into image during diffusion
    """
        Args:
        beta_start: The starting beta for sampler noise levels.

        beta_end: The final beta for sampler noise levels.

        beta_schedule: The schedule to use for beta.
            Can be 'scaled_linear', 'scaled_cosine', 'scaled_sqrt' or 
            'linear', 'cosine', or 'sqrt' which uses the unscaled function.

        clip_sample: Whether to clip the sampler noise level when
            sampling. Should be disabled for best performance with DDPM.
            
        set_alpha_to_one: Set alpha to exactly 1. 
    """
    scheduler = DDIMScheduler(beta_start=0.00085 , 
                            beta_end= 0.012 ,
                            beta_schedule="scaled_linear",
                            clip_sample= False ,
                            set_alpha_to_one=False)
    
    return (Vae,UNet,tokenizer,text_encoder,scheduler)

def image_to_latents(img):
    """
    function to convert the input image to latents from perform 
    diffusion process.
    """
    # conversion of image to tensor 
    init_img = tfms.ToTensor()(img).unsqueeze(0)

    # normalize to convert range from (0,1) --> (-1,1)
    init_img = (init_img * 2.0) - 1.0 

    # now the image tensor is loaded into GPU from parallel computation
    init_img = init_img.to(device='cuda', dtype=torch.float16)
    
    # converting image tensor into latent representation using VAE
    img_latent = Vae.encode(init_img).latent_dist.sample() * 0.18214 # scaling term

    return img_latent



def latent_to_image(latents):
    """
    function to convert latent representation back to image 
    """
    # scaling back to original range
    latents = (1/0.18215) * latents

    # decoding the latents
    with torch.no_grad():
        image = Vae.decode(latents).sample

    # Normailze the images.
    image = (image/2 + 0.5).clamp(0,1)

    # Convert the images to numpy array 
    image = image.detach().cpu().permute(0,2,3,1).numpy()

    # scaling to tensor value back to pixel values ( 0 to 255 )
    image = (image * 255).round().astype('uint8')

    #convert the numpy array to a list of PIL images 
    pil_images = [Image.fromarray(img) for img in image ]

    return pil_images



def text_embeddings(prompts , max_length = None):
    """
    function to convert text prompt to vector embeddings 
    """

    if max_length is None :
        max_length = tokenizer.model_max_length

    # converting text prompt into tokens 
    tokens = tokenizer(prompts , padding = "max_length",
                    max_length = max_length,
                    trancation= True , return_tensors = "pt")
    
    # converting tokens into input ids with half precision 
    encoded_text = text_encoder(tokens.input_ids.to("cuda"))[0].half()

    return encoded_text


Vae,UNet,tokenizer,text_encoder,scheduler  = load_artifacts() 

    

