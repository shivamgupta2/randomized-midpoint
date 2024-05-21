from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler

from typing import List, Tuple, Union, Optional
import torch
from diffusers.utils.torch_utils import randn_tensor

class RandomizedMidpointScheduler(DDIMScheduler):

    def _get_variance(self, timestep, prev_timestep, get_randomized_midpoint=False):
        device = self.device
        batch_size = self.batch_size

        alpha_prod_t = self.alphas_cumprod[timestep]
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        if get_randomized_midpoint:
            alpha_prod_t_prev = torch.zeros(batch_size, device=device)
            alpha_prod_t_prev[prev_timestep >= 0] = self.alphas_cumprod[prev_timestep[prev_timestep >= 0]]
            alpha_prod_t_prev[prev_timestep < 0] = self.final_alpha_cumprod
        else: 
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        get_randomized_midpoint=False,
        get_deterministic_midpoint=False,
        randomized_midpoint_second_step=False,
        randomized_midpoint_exp_alpha_h=None,
        randomized_midpoint_beta_prod_t=None
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Get randomized midpoint between this step and the previous step. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        self.batch_size = len(model_output)
        batch_size = self.batch_size
        self.device = model_output.device
        device = self.device

        if get_randomized_midpoint:
            #compute random midpoint location (between 0 and 1)
            midpoint_loc = torch.rand(batch_size, device=model_output.device)

            # 1. get previous step value (=t-1)
            prev_timestep = timestep - (2 * midpoint_loc * (self.config.num_train_timesteps // self.num_inference_steps)).long()
        elif randomized_midpoint_second_step:
            prev_timestep = timestep - 2 * (self.config.num_train_timesteps//self.num_inference_steps)
        elif get_deterministic_midpoint:
            midpoint_loc = torch.ones(batch_size, device = model_output.device) * 0.5
            prev_timestep = timestep - (midpoint_loc * (self.config.num_train_timesteps/self.num_inference_steps)).long()
        else:
            # 1. get previous step value (=t-1)
            prev_timestep = timestep - self.config.num_train_timesteps//self.num_inference_steps
            step_size = self.config.num_train_timesteps//self.num_inference_steps


        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)
        if get_randomized_midpoint or get_deterministic_midpoint:
            alpha_prod_t_prev = torch.zeros(batch_size, device=model_output.device)
            alpha_prod_t_prev[prev_timestep >= 0] = self.alphas_cumprod[prev_timestep[prev_timestep >= 0]]
            alpha_prod_t_prev[prev_timestep < 0] = self.final_alpha_cumprod
        else: 
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod


        beta_prod_t = 1 - alpha_prod_t

        print(self.config.prediction_type)

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        if randomized_midpoint_second_step:
            score_est = -pred_epsilon/(randomized_midpoint_beta_prod_t ** (0.5))
        else:
            score_est = -pred_epsilon/(beta_prod_t ** (0.5))

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep, get_randomized_midpoint=get_randomized_midpoint or get_deterministic_midpoint)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if get_randomized_midpoint or get_deterministic_midpoint:
            print('first step')
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5)
            pred_sample_direction = torch.einsum('i,ijkl->ijkl', pred_sample_direction, pred_epsilon)

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            prev_sample = torch.einsum('i,ijkl->ijkl', alpha_prod_t_prev ** (0.5), pred_original_sample) + pred_sample_direction


            #h_est = torch.log((alpha_prod_t_prev/alpha_prod_t)**(0.5))
            #prev_sample = torch.einsum('i,ijkl->ijkl', (alpha_prod_t_prev/alpha_prod_t)**(0.5), sample) + torch.einsum('i,ijkl->ijkl', ((alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5)) - 1), score_est)
        elif randomized_midpoint_second_step:
            print('second step')
            #h_est = torch.log((alpha_prod_t_prev/alpha_prod_t)**(0.5))
            h_est = torch.log((alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5)))
            #prev_sample = (alpha_prod_t_prev/alpha_prod_t) ** (0.5) * sample + torch.einsum('i,ijkl->ijkl', h_est * torch.exp((1 - randomized_midpoint_locs) * h_est), score_est)
            prev_sample = (alpha_prod_t_prev/alpha_prod_t) ** (0.5) * sample + torch.einsum('i,ijkl->ijkl', h_est * ((alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5)))/randomized_midpoint_exp_alpha_h, score_est)
        else:
            #DDIM
            # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            #Just use calculation from 1c
            #prev_sample = (alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5)) * sample + ((alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5)) - 1) * score_est

            #Just use calculation from 1d
            #h_est = torch.log((alpha_prod_t_prev/alpha_prod_t)**(0.5))
            #prev_sample = (alpha_prod_t_prev/alpha_prod_t) ** (0.5) * sample +  h_est * ((alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5))) * score_est


        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            if get_randomized_midpoint:
                return prev_sample, prev_timestep.to(model_output.device), (alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5)), beta_prod_t
            else:
                return prev_sample

        if get_randomized_midpoint:
            #return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample), prev_timestep.to(model_output.device), midpoint_loc
            return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample), prev_timestep.to(model_output.device), (alpha_prod_t_prev ** (0.5))/(alpha_prod_t ** (0.5)), beta_prod_t
        else:
            return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample), prev_timestep.to(model_output.device)
