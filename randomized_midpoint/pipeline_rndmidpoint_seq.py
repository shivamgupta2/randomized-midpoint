# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch

from diffusers import DDIMScheduler
#from utils.torch_utils import randn_tensor
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, ImagePipelineOutput



@torch.no_grad()
def randomized_midpoint_forward(
    self,
    batch_size: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    eta: float = 0.0,
    num_inference_steps: int = 50,
    use_clipped_model_output: Optional[bool] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    use_randomized_midpoint=False,
    latents = None
) -> Union[ImagePipelineOutput, Tuple]:
    r"""
    The call function to the pipeline for generation.

    Args:
        batch_size (`int`, *optional*, defaults to 1):
            The number of images to generate.
        generator (`torch.Generator`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
            DDIM and `1` corresponds to DDPM.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        use_clipped_model_output (`bool`, *optional*, defaults to `None`):
            If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
            downstream to the scheduler (use `None` for schedulers which don't support this argument).
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

    Example:

    ```py
    >>> from diffusers import DDIMPipeline
    >>> import PIL.Image
    >>> import numpy as np

    >>> # load model and scheduler
    >>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

    >>> # run pipeline in inference (sample random noise and denoise)
    >>> image = pipe(eta=0.0, num_inference_steps=50)

    >>> # process image to PIL
    >>> image_processed = image.cpu().permute(0, 2, 3, 1)
    >>> image_processed = (image_processed + 1.0) * 127.5
    >>> image_processed = image_processed.numpy().astype(np.uint8)
    >>> image_pil = PIL.Image.fromarray(image_processed[0])

    >>> # save image
    >>> image_pil.save("test.png")
    ```

    Returns:
        [`~pipelines.ImagePipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
            returned where the first element is a list with the generated images
    """

    print('Randomized Midpoint!', flush=True)

    # Sample gaussian noise to begin loop
    if isinstance(self.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            self.unet.config.in_channels,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is not None:
        image = latents
    else:
        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

    # set step values
    self.scheduler.set_timesteps(num_inference_steps)

    used_randomized_midpoint_flag = False
    for t in self.progress_bar(self.scheduler.timesteps):
        if used_randomized_midpoint_flag:
            used_randomized_midpoint_flag = False
            continue
        if t < 1000:
            use_randomized_midpoint = False
        # 1. predict noise model_output
        model_output = self.unet(image, t).sample

        if use_randomized_midpoint:
            randomized_midpoint_output, midpoint_timestep, randomized_midpoint_exp_alpha_h, randomized_midpoint_beta_prod_t = self.scheduler.step(model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator, get_randomized_midpoint=True)
            randomized_midpoint = randomized_midpoint_output.prev_sample

            model_output = self.unet(randomized_midpoint.half(), midpoint_timestep).sample
            image, _ = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator, randomized_midpoint_second_step = True, randomized_midpoint_exp_alpha_h=randomized_midpoint_exp_alpha_h, randomized_midpoint_beta_prod_t = randomized_midpoint_beta_prod_t)
            used_randomized_midpoint_flag = True
            #image, _ = self.scheduler.step(
                    #model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator)
        # 2. predict previous mean of image x_t-1 and add variance depending on eta
        # eta corresponds to η in paper and should be between [0, 1]
        # do x_t -> x_t-1
        else:
            image, _ = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            )
        image = image.prev_sample.half()

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image,)

    return ImagePipelineOutput(images=image)
