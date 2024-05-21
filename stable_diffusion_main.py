import torch
import imageio
import types
import os

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMPipeline
from randomized_midpoint.pipeline_rndmidpoint_seq import randomized_midpoint_forward
from randomized_midpoint.scheduler_rndmidpoint_seq import RandomizedMidpointScheduler
from randomized_midpoint.pipeline_stablediffusion_rndmidpoint_seq import stable_diffusion_randomized_midpoint_forward


def run(rank):
    prompt = "a photograph of an astronaut riding a horse"
    model_str = 'CompVis/stable-diffusion-v1-4'
    model_label = 'stable_diffusion'
    inference_steps_list = [8, 16, 24, 32, 40]

    # Initialize the first pipeline (DDIM)
    pipe_ddim = StableDiffusionPipeline.from_pretrained(model_str, revision="fp16", torch_dtype=torch.float16)
    pipe_ddim = pipe_ddim.to(f'cuda:{rank}')
    pipe_ddim.scheduler.timestep_scaling = 'trailing'
    pipe_ddim.unet.eval()

    # Initialize the second pipeline with RandomizedMidpointScheduler
    pipe_rnd = StableDiffusionPipeline.from_pretrained(model_str, revision="fp16", torch_dtype=torch.float16)
    pipe_rnd.scheduler = RandomizedMidpointScheduler.from_config(pipe_rnd.scheduler.config)
    pipe_rnd.scheduler.timestep_scaling = 'trailing'
    pipe_rnd.randomized_midpoint_forward = types.MethodType(stable_diffusion_randomized_midpoint_forward, pipe_rnd)
    pipe_rnd = pipe_rnd.to(f'cuda:{rank}')
    pipe_rnd.unet.eval()

    total_num_images = 1
    batch_size = 1

    # Generate latents
    seed = 26
    generator = torch.Generator(device=f'cuda:{rank}').manual_seed(seed)
    latents = torch.randn((batch_size, pipe_ddim.unet.config.in_channels, 64, 64), generator=generator, device=f'cuda:{rank}', dtype=torch.float16)

    for num_inference_steps in inference_steps_list:
        # Generate image with the first pipeline (DDIM)
        ddim_out_dir = f'results/ddim_{model_label}'
        os.makedirs(ddim_out_dir, exist_ok=True)
        images_ddim = pipe_ddim(prompt, latents=latents, num_inference_steps=num_inference_steps).images
        image_ddim = images_ddim[0]
        image_path_ddim = os.path.join(ddim_out_dir, f'ddim_output_{num_inference_steps:02d}_steps.png')
        image_ddim.save(image_path_ddim)

        # Generate image with the second pipeline (RandomizedMidpointScheduler)
        rnd_midpoint_out_dir = f'results/randomized_midpoint_stable'
        os.makedirs(rnd_midpoint_out_dir, exist_ok=True)
        images_rnd = pipe_rnd.randomized_midpoint_forward(prompt, latents=latents, num_inference_steps=num_inference_steps, use_randomized_midpoint=True).images
        image_rnd = images_rnd[0]
        image_path_rnd = os.path.join(rnd_midpoint_out_dir, f'stable_diffusion_rnd_midpoint_{num_inference_steps:02d}_steps.png')
        image_rnd.save(image_path_rnd)

        #det_midpoint_out_dir = f'results/deterministic_midpoint_stable'
        #os.makedirs(det_midpoint_out_dir, exist_ok=True)
        #images_det = pipe_rnd.randomized_midpoint_forward(prompt, latents=latents, num_inference_steps=num_inference_steps//2, use_deterministic_midpoint=True).images
        #image_det = images_det[0]
        #image_path_det = os.path.join(det_midpoint_out_dir, f'stable_diffusion_det_midpoint_{num_inference_steps:02d}_steps.png')
        #image_det.save(image_path_det)


def main():
    torch.autograd.set_detect_anomaly(True)
    run(3)

if __name__ == '__main__':
    main()

