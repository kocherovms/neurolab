import torch
import torchvision.transforms.functional as VF

class ObPreprocessor:
    def __init__(self, ob_shape, cast_to_float=False):
        self.ob_shape = ob_shape
        self.cast_to_float = cast_to_float
        
    def __call__(self, obs):
        assert isinstance(obs, torch.Tensor)
        assert obs.dtype == torch.uint8
        obs_ndim = obs.ndim
        assert obs_ndim >= 3 # height, width, color
    
        if obs_ndim == 3:
            obs = obs.unsqueeze(0) # ensure there is a batch dim

        obs_shape = obs.shape
        obs = obs.view(-1, *obs_shape[-3:])

        if obs.shape[-3:] == (210, 160, 3):
            # Default interleaved layout from ALE
            obs = obs.permute(0, 3, 1, 2) # move color channel to the front (switch interleaved->planar format)
        elif obs.shape[-3:] == (3, 210, 160):
            # Planar layout preprocessed by dataset's RolloutManager
            pass
        else:
            assert False, f'Unsupported {obs.shape[-3:]=}'

        if self.ob_shape == (3, 210, 160):
            pass
        elif self.ob_shape == (1, 178, 152):
            obs = VF.rgb_to_grayscale(obs, num_output_channels=1)
            obs = obs[:,:,8:186,8:160] # extract only meaningful data for Frostbite
            obs = obs.view(*obs_shape[:-3], 1, 178, 152)
        elif self.ob_shape == (3, 178, 152):
            obs = obs[:,:,8:186,8:160] # extract only meaningful data for Frostbite
            obs = obs.view(*obs_shape[:-3], 3, 178, 152)
        elif self.ob_shape == (3, 89, 76): # scale down by 2 original shape (178, 152)
            obs = obs[:,:,8:186,8:160] # extract only meaningful data for Frostbite
            obs = VF.resize(obs, [89, 76], interpolation=VF.InterpolationMode.BILINEAR, antialias=True)
            obs = obs.view(*obs_shape[:-3], 3, 89, 76)
        elif self.ob_shape == (3, 84, 84):
            obs = VF.resize(obs, [84, 84], interpolation=VF.InterpolationMode.BILINEAR, antialias=True)
            obs = obs.view(*obs_shape[:-3], 3, 84, 84)
        elif self.ob_shape == (1, 84, 84):
            obs = VF.rgb_to_grayscale(obs, num_output_channels=1)
            obs = VF.resize(obs, [84, 84], interpolation=VF.InterpolationMode.BILINEAR, antialias=True)
            obs = obs.view(*obs_shape[:-3], 1, 84, 84)
        else:
            assert False, f'Unsupported {self.ob_shape=}'
        
        if obs_ndim == 3:
            obs = obs.squeeze(0)

        if self.cast_to_float:
            return obs / 255.0
            
        return obs
        