import torch
from utils import text_embeddings , scheduler ,image_to_latents ,UNet
from tqdm.auto import tqdm
import numpy as np

def prompt_to_img_i2i(prompts, init_img,
                    neg_prompt = None, g = 7.5,
                    seed = 100 ,strength = 0.8 ,
                    steps = 50 , dim = 512):
    """
    function to perform diffusion process to convert prompt into image 
    """

    # converting text prompt into embedding
    encoded_text = text_embeddings(prompts)

    # adding an unconditional prompt which helps in the generation process
    if not neg_prompt : uncond = text_embeddings([""], text.shape[1])
    else : uncond = text_embeddings(neg_prompt , text.shape[1])
    embeddings = torch.cat([uncond,encoded_text])

    # setting up the seed 
    if seed : torch.manual_seed(seed)

    # setting the number of steps in scheduler
    scheduler.set_timesteps(steps)

    # convert the initial image into latents 
    init_latents = image_to_latents(init_img)

    # setting up the initial time step based on the strength 
    init_timestep = int(steps * strength)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] , device = 'cuda')

    # adding noise to the latents 
    noise = torch.randn(init_latents.shape , generator = None ,
                        device= 'cuda', dtype = init_latents.dtype)
    latents = scheduler.add_noise(init_latents , noise , timesteps)

    # computing the starting timestep to start the diffusion process
    starting_timestep = max(steps - init_timestep , 0 )
    timesteps = scheduler.timesteps[starting_timestep:].to('cuda')

    for i , timestep in enumerate(tqdm(timesteps)):

        # scaling the latents according to the timestep to match the variance 
        scaled_latents = scheduler.scale_model_input(torch.cat([latents] * 2),timestep)

        # Predicting the noise residual using Unet
        with torch.no_grad():
            u , t = UNet( scaled_latents , timestep,
                        encoder_hidden_states = embeddings).sample.chunk(2)

        # performing guidance 
        pred = u + g*(t-u)

        # updating the latents based on the guidance,noise residual and current timestep
        final_latents = scheduler.step(pred , timestep, latents).prev_sample
    

    return final_latents.detach().cpu()




def create_mask ( init_img ,refer_prompt , 
                  query_prompt , n=10 ,s=0.5):
    
    """
    function to create a mask over an input image 
    """

    # dictionary to save the difference of reference denoised sample and query denoised sample
    diff = {}

    for idx  in range(n):
        
        # creating a denoised sample using reference prompt
        refer_sample = prompt_to_img_i2i(prompts = refer_prompt,init_img= init_img,
                                    strength= s , seed=100* idx)[0]

        # creating a denoised sample using query prompt
        query_sample = prompt_to_img_i2i(prompts = query_prompt , init_img=init_img,
                                        strength= s ,seed = 100*idx)[0]

        # taking the difference of both sample
        diff[idx] = (np.array(refer_sample) - np.array(query_sample))

    
    # creating a mask placeholder 
    mask = np.zeros_like(diff[0])

    # taking the average of n iterations 
    for idx in range(n):
        mask += np.abs(diff[idx])

    #averaging multiple channels 
    mask = mask.mean(0)

    # normalizing the mask
    mask = ( mask - mask.mean()) / np.std(mask)

    # binarzing and returning the mask object
    return ( mask > 0).astype('uint8')






