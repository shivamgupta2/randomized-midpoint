import torch
import imageio
import types
import os

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMPipeline
from randomized_midpoint.pipeline_rndmidpoint_seq import randomized_midpoint_forward
from randomized_midpoint.scheduler_rndmidpoint_seq import RandomizedMidpointScheduler
from torchvision import transforms


def run(rank):
    #model_str = 'stabilityai/stable-diffusion-2'
    #prompts=['beautiful castle, matte painting']

    #model_str = 'google/ddpm-cifar10-32'
    #model_label='cifar'

    #model_str = 'google/ddpm-ema-celebahq-256'
    #model_label = 'celeba'

    #model_str = 'google/ddpm-bedroom-256'
    #model_label='bedroom'

    #model_str = 'fusing/ddpm-cifar10-ema'

    #model_str = 'CompVis/stable-diffusion-v1-4'
    #model_label = 'stable_diffusion'

    model_str = './ddpm_ema_cifar10'
    model_label = 'cifar_ema'

    #Initialize the DDIM pipeline
    pipe_ddim = DDIMPipeline.from_pretrained(model_str, torch_dtype=torch.float16)
    pipe_ddim = pipe_ddim.to(f'cuda:{rank}')
    pipe_ddim.scheduler = RandomizedMidpointScheduler.from_config(pipe_ddim.scheduler.config)
    pipe_ddim.randomized_midpoint_forward = types.MethodType(randomized_midpoint_forward, pipe_ddim)
    pipe_ddim.scheduler.timestep_scaling = 'trailing'
    pipe_ddim.unet.eval()

    #Initialize the Randomized Midpoint pipeline
    pipe_rnd = DDIMPipeline.from_pretrained(model_str, torch_dtype=torch.float16)
    pipe_rnd.to(f'cuda:{rank}')
    pipe_rnd.scheduler = RandomizedMidpointScheduler.from_config(pipe_rnd.scheduler.config)
    pipe_rnd.randomized_midpoint_forward = types.MethodType(randomized_midpoint_forward, pipe_rnd)
    pipe_rnd.scheduler.timestep_scaling = 'trailing'
    pipe_rnd.unet.eval()


    #scheduler = RandomizedMidpointScheduler.from_pretrained(model_str, subfolder='scheduler', timestep_scaling='trailing')
    #scheduler._is_ode_scheduler = True
    #nfes_list = [6, 20, 30, 40, 50, 100, 200, 500, 1000]
    #nfes_list = [6, 10, 20]
    #nfes_list = [10]
    #nfes_list = [200, 500, 1000]
    #nfes_list = [6, 12, 18, 24, 30]
    #nfes_list = [40, 50, 60, 70]
    nfes_list = [10]
    long_nfes = 1000

    total_num_images = 50000
    batch_size = 4096

    seed = 34
    generator = torch.Generator(device=f'cuda:{rank}').manual_seed(seed)
    ddim_error = 0
    randomized_midpoint_error = 0
    for ind in range(0, total_num_images, batch_size):
        latents = torch.randn((batch_size, pipe_ddim.unet.config.in_channels, 32, 32), generator = generator, device=f'cuda:{rank}', dtype=torch.float16)
        #latents = torch.randn((batch_size, pipe_ddim.unet.config.in_channels, 256, 256), generator = generator, device=f'cuda:{rank}', dtype=torch.float16)

        #ode_solution_out_dir = f'results/ode_{model_label}'
        #os.makedirs(ode_solution_out_dir, exist_ok=True)
        #images_ode = pipe_ddim.randomized_midpoint_forward(latents=latents, num_inference_steps=long_nfes, use_randomized_midpoint=False).images
        #image_ode = images_ode[0]
        #image_path_ode = os.path.join(ode_solution_out_dir, f'ode_output_{long_nfes:03d}_steps.png')
        #image_ode.save(image_path_ode)

        for nfes in nfes_list:
            #DDIM output
            ddim_out_dir = f'results/EI_ddim_{model_label}_nfes_{nfes}'
            os.makedirs(ddim_out_dir, exist_ok=True)
            images_ddim = pipe_ddim.randomized_midpoint_forward(latents=latents, num_inference_steps=nfes, use_randomized_midpoint=False).images
            for i, image_ddim in enumerate(images_ddim):
                image_path_ddim = os.path.join(ddim_out_dir, f'ddim_output_{nfes:03d}_steps_{ind + i}.png')
                image_ddim.save(image_path_ddim)

            #Randomized midpoint output
            rnd_midpoint_out_dir = f'results/randomized_midpoint_{model_label}_nfes_{nfes}'
            os.makedirs(rnd_midpoint_out_dir, exist_ok=True)
            images_rnd = pipe_rnd.randomized_midpoint_forward(latents=latents, num_inference_steps=nfes, use_randomized_midpoint=True).images
            for i, image_rnd in enumerate(images_rnd):
                image_path_rnd = os.path.join(rnd_midpoint_out_dir, f'rnd_midpoint_{nfes:03d}_steps_{ind + i}.png')
                image_rnd.save(image_path_rnd)
    print('done with images')
        
        #for i in range(len(images_ode)):
        #    image_ode_torch = transforms.ToTensor()(images_ode[i])
        #    image_ddim_torch = transforms.ToTensor()(images_ddim[i])
        #    image_rnd_torch = transforms.ToTensor()(images_rnd[i])
        #    ddim_error += torch.norm(image_ode_torch - image_ddim_torch)
        #    randomized_midpoint_error += torch.norm(image_ode_torch - image_rnd_torch)
    #ddim_error /= total_num_images
    #randomized_midpoint_error /= total_num_images
    #print('DDIM error:', ddim_error)
    #print('Randomized Midpoint error:', randomized_midpoint_error)



    #for nfes in nfes_list:
    #    rnd_midpoint_out_dir = f'results/randomized_midpoint_nfes:{nfes:06d}_{model_label}'
    #    try:
    #        os.makedirs(rnd_midpoint_out_dir)
    #    except OSError as e:
    #        pass

    #    for ind in range(0, total_num_images, batch_size):
    #        images = pipe.randomized_midpoint_forward(eta=0.0, num_inference_steps=nfes//2, batch_size=batch_size, use_randomized_midpoint=True).images
    #        for batch_ind in range(0, batch_size):
    #            if ind + batch_ind < total_num_images:
    #                image_path = os.path.join(rnd_midpoint_out_dir, f'rnd_midpoint_{(ind+batch_ind):06d}.png')
    #                images[batch_ind].save(image_path)

    #pipe = StableDiffusionPipeline.from_pretrained(model_str, torch_dtype=torch.float16)
    #pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    #print(pipe.scheduler.config)

    #pipe = pipe.to(f'cuda:{rank}')
    #image = pipe(prompts[0]).images[0]

def main():
    torch.autograd.set_detect_anomaly(True)
    run(3)

if __name__ == '__main__':
    main()
