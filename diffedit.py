from utils import text_embeddings,scheduler , image_to_latents , UNet,latent_to_image
import torch
from tqdm.auto import tqdm



def prompt_to_img_diffedit(init_img , refer_prompt, query_prompt,
                        mask ,g = 7.5 , seed = 100 , strength =0.7,
                        steps = 70 , dim = 512 ): 
    
    # coverting prompts into encode text 
    refer_encoded_text = text_embeddings(refer_prompt)
    query_encoded_text = text_embeddings(query_prompt)

    # Adding Unconditional prompt 
    uncond = text_embeddings([""],refer_encoded_text.shape[1])
    embeddings = torch.cat([uncond , refer_encoded_text, query_encoded_text])


    ## setting the seed 
    if seed: torch.manual_seed(seed)

    # setting up the number of steps in scheduler 
    scheduler.set_timesteps(steps)

    # converting the initial image in latents 
    init_latents = image_to_latents(init_img)

    # setting up the initial timestep based on strength
    init_timestep = int(steps * strength)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device = 'cuda')

    # adding noise into image latent
    noise = torch.randn(init_latents.shape , generator= None 
                        ,device= 'cuda' ,dtype = init_latents.dtype)
    latents = scheduler.add_noise(init_latents , noise , timesteps)

    # computing the starting timestep to start the diffusion process
    starting_timestep = max(steps - init_timestep , 0)
    timesteps = scheduler.timesteps[starting_timestep:].to('cuda')

    mask = torch.tensor(mask , dtype= UNet.dtype).unsqueeze(0).unsqueeze(0).to('cuda')

    for i , timestep in enumerate(tqdm(timesteps)):
        
        # scaling the latents according to the timestep to match the variance 
        scaled_latents = scheduler.scale_model_input(torch.cat([latents] * 3),timestep)

        #predicting the noise residual using Unet
        with torch.no_grad():
            u , rt , qt = UNet(scaled_latents , timestep , 
                               encoder_hidden_states = embeddings).sample.chunk(3)
        
        # performing guidance 
        refer_pred = u + g(rt - u)
        query_pred = u + g(qt - u)

        # updating the latent based the timestep 
        refer_latents = scheduler.step(refer_pred ,timestep ,latents).prev_sample
        query_latents = scheduler.step(query_pred ,timestep ,latents).prev_sample
        final_latents = mask * query_latents + (1 - mask) * refer_latents
    

    return latent_to_image(final_latents)