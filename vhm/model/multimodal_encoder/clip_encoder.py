import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from .configuration_evaclip import EvaCLIPVisionConfig
from .modeling_evaclip import EvaCLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(
            args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(
                self.vision_tower_name)

    def load_model(self):
        print(f'Load vision tower from {self.vision_tower_name}')
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name)
        if 'eva' in self.vision_tower_name.lower():
            vision_cfg = EvaCLIPVisionConfig.from_pretrained(
                self.vision_tower_name)
            self.vision_tower = EvaCLIPVisionModel.from_pretrained(
                self.vision_tower_name, config=vision_cfg)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if len(self.select_layer)==1:
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(
                    f'Unexpected select feature: {self.select_feature}')
        else:
            if self.select_feature == 'patch':
                image_features = torch.cat([image_forward_outs.hidden_states[layer][:, 1:] for layer in self.select_layer], dim=-1)
            elif self.select_feature == 'cls_patch':
                image_features = torch.cat([image_forward_outs.hidden_states[layer] for layer in self.select_layer], dim=-1)
            else:
                raise ValueError(
                    f'Unexpected select feature: {self.select_feature}')

        # raise ValueError(image_features.size())
        return image_features

    # @torch.no_grad() comment to enable fine-tune vit
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(
                    device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(
                    image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(
                image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
