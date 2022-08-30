import torch.cuda

from VAE_test import VAEWrapper
import crafter
import gym
import torch

env = gym.make("CrafterReward-v1")
print(env.action_space)
print(torch.cuda.is_available())
kwargs = {
    "channel_last": True,
    "image_channels": 3,
    "latent_size": 130,
    "h_dim": 768,
    "obs_label": "latent_obs",
    "conv_net": True,
    "steps": 250,
    "step_skip": 10,
    "vae" : None
}
vae = VAEWrapper(env=env, eval=True, device=None, **kwargs)

i = 0
vae.reset()
while i < 25:
    obs, rew, done, info = vae.step(12)
    print(obs)
