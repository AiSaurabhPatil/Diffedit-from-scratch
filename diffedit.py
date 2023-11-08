from utils import tokenizer,scheduler , pil_to_latents , unet,latents_to_pil
import torch
from tqdm.auto import tqdm
from utils import text_enc , tokenizer,pil_to_latents , unet , UNet2DConditionModel , latents_to_pil

def prompt_2_img_i2i(prompts, init_img, neg_prompts=None, g=7.5, seed=100, strength =0.8, steps=50, dim=512):
    """
    Diffusion process to convert prompt to image
    """
    # Converting textual prompts to embedding
    text = text_enc(prompts) 
    
    # Adding an unconditional prompt , helps in the generation process
    if not neg_prompts: uncond =  text_enc([""], text.shape[1])
    else: uncond =  text_enc(neg_prompt, text.shape[1])
    emb = torch.cat([uncond, text])
    
    # Setting the seed
    if seed: torch.manual_seed(seed)
    
    # Setting number of steps in scheduler
    scheduler.set_timesteps(steps)
    
    # Convert the seed image to latent
    init_latents = pil_to_latents(init_img)
    
    # Figuring initial time step based on strength
    init_timestep = int(steps * strength) 
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device="cuda")
    
    # Adding noise to the latents 
    noise = torch.randn(init_latents.shape, generator=None, device="cuda", dtype=init_latents.dtype)
    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    latents = init_latents
    
    # Computing the timestep to start the diffusion loop
    t_start = max(steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:].to("cuda")
    
    # Iterating through defined steps
    for i,ts in enumerate(tqdm(timesteps)):
        # We need to scale the i/p latents to match the variance
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
        
        # Predicting noise residual using U-Net
        with torch.no_grad(): u,t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
            
        # Performing Guidance
        pred = u + g*(t-u)

        # Conditioning  the latents
        #latents = scheduler.step(pred, ts, latents).pred_original_sample
        latents = scheduler.step(pred, ts, latents).prev_sample
    
    # Returning the latent representation to output an array of 4x64x64
    return latents.detach().cpu()

def prompt_2_img_diffedit(rp, qp, init_img, mask, g=7.5, seed=100, strength =0.7, steps=70, dim=512):
    """
    Diffusion process to convert prompt to image
    """
    # Converting textual prompts to embedding
    rtext = text_enc(rp) 
    qtext = text_enc(qp)
    
    # Adding an unconditional prompt , helps in the generation process
    uncond =  text_enc([""], rtext.shape[1])
    emb = torch.cat([uncond, rtext, qtext])
    
    # Setting the seed
    if seed: torch.manual_seed(seed)
    
    # Setting number of steps in scheduler
    scheduler.set_timesteps(steps)
    
    # Convert the seed image to latent
    init_latents = pil_to_latents(init_img)
    
    # Figuring initial time step based on strength
    init_timestep = int(steps * strength) 
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device="cuda")
    
    # Adding noise to the latents 
    noise = torch.randn(init_latents.shape, generator=None, device="cuda", dtype=init_latents.dtype)
    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    latents = init_latents
    
    # Computing the timestep to start the diffusion loop
    t_start = max(steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:].to("cuda")
    
    # Converting mask to torch tensor
    mask = torch.tensor(mask, dtype=unet.dtype).unsqueeze(0).unsqueeze(0).to("cuda")
    
    # Iterating through defined steps
    for i,ts in enumerate(tqdm(timesteps)):
        # We need to scale the i/p latents to match the variance
        inp = scheduler.scale_model_input(torch.cat([latents] * 3), ts)
        
        # Predicting noise residual using U-Net
        with torch.no_grad(): u, rt, qt = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(3)
            
        # Performing Guidance
        rpred = u + g*(rt-u)
        qpred = u + g*(qt-u)

        # Conditioning  the latents
        rlatents = scheduler.step(rpred, ts, latents).prev_sample
        qlatents = scheduler.step(qpred, ts, latents).prev_sample
        latents = mask*qlatents + (1-mask)*rlatents
    
    # Returning the latent representation to output an array of 4x64x64
    return latents_to_pil(latents)
