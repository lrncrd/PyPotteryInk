import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        
        # Define target modules for both new training and checkpoint loading
        self.target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        self.target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae

        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        # Initialize VAE and UNet
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(self.vae.encoder, self.vae.encoder.__class__)
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(self.vae.decoder, self.vae.decoder.__class__)
        
        # Add skip connections
        self.vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.vae.decoder.ignore_skip = False

        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # Move models to GPU
        self.vae.to("cuda")
        self.unet.to("cuda")

        # If loading from checkpoint
        if pretrained_path is not None and pretrained_path.endswith('.pkl'):
            print(f"Loading checkpoint from {pretrained_path}")
            sd = torch.load(pretrained_path, map_location="cpu")
            
            # Store the configuration from checkpoint
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            self.target_modules_unet = sd["unet_lora_target_modules"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            
            # Load LoRA configurations
            unet_lora_config = LoraConfig(
                r=self.lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_unet
            )
            vae_lora_config = LoraConfig(
                r=self.lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_vae
            )
            
            # Add adapters and load weights
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            self.unet.add_adapter(unet_lora_config)
            
            # Load state dictionaries
            _sd_vae = self.vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            self.vae.load_state_dict(_sd_vae)
            
            _sd_unet = self.unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_unet)
            
            print("Checkpoint loaded successfully")
        
        # For new training
        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(self.vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(self.vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(self.vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(self.vae.decoder.skip_conv_4.weight, 1e-5)
            
            vae_lora_config = LoraConfig(
                r=self.lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_vae
            )
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            
            unet_lora_config = LoraConfig(
                r=self.lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_unet
            )
            self.unet.add_adapter(unet_lora_config)

        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        # unet.enable_xformers_memory_efficient_attention()
        self.vae.decoder.gamma = 1

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        """Save the model checkpoint"""
        sd = {
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "state_dict_unet": {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k},
            "state_dict_vae": {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        }
        torch.save(sd, outf)