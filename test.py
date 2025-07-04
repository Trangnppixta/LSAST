import argparse, os, sys, glob
import PIL
import cv2
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
import einops
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from transformers import CLIPProcessor, CLIPModel
from cldm.model import create_model, load_state_dict
import re

seed = 42
strength = 0.8
ddim_steps = 50
low_threshold = 100
high_threshold = 200
# prompt = 'a brown short-haired woman'
# contentdir = "data/test_data/woman1.jpg"
contentdir = "/mnt/md0/projects/ai_headshot/style_transfer_data/content-image/123753_3/w2.jpg"
embed_dir = "logs/train2025-06-30-08-01-09_v1-finetune/testtube/version_0/checkpoints/embeddings_gs-26999.pt"
n_samples = 1

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)
apply_canny = CannyDetector()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512,512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/sd1.5/v1-5-pruned.ckpt"
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")
sampler = DDIMSampler(model)
canny_model_path = "./models/controlnet/control_v11p_sd15_canny.pth"


def main(prompt='', content_dir=None, ddim_steps=50, strength=0.5, ddim_eta=0.0, n_iter=1, C=4, f=8, n_rows=0,
         scale=10.0, \
         model=None, seed=42, prospect_words=None, n_samples=1, height=512, width=512):
    precision = "autocast"
    outdir = "outputs/comparison"
    seed_everything(seed)
    controlnet_canny = create_model(
        './configs/controlnet/control_canny.yaml')
    controlnet_canny = controlnet_canny.to(device)
    model_resume_state = torch.load(canny_model_path, map_location='cpu')
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) + 10

    if content_dir is not None:
        content_name = content_dir.split('/')[-1].split('.')[0]
        content_image = load_img(content_dir).to(device)
        content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
        content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

        init_latent = content_latent

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")



    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])                            
                            
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c = model.get_learned_conditioning(prompts, prospect_words=prospect_words)
                        img = cv2.imread(content_dir)
                        img = cv2.resize(img, (512, 512))
                        # H, W, C = img.shape
                        detected_map = apply_canny(img, low_threshold, high_threshold)
                        name = content_name + '-' + str(low_threshold) + '-' + str(high_threshold) + '.png'
                        cv2.imwrite(name, detected_map)
                        image = detected_map[:, :, None]
                        image = np.concatenate([image, image, image], axis=2)
                        # img1 = rearrange(detected_map, 'h w c ->c h w')

                        control = torch.from_numpy(image.copy()).float().to(device) / 255.0

                        control = torch.stack([control for _ in range(n_samples)], dim=0)
                        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                        new_state_dict = {}
                        for key, value in model_resume_state.items():
                            if key.startswith('control_model.'):
                                new_key = key[len('control_model.'):]  # 去掉前缀
                                new_state_dict[new_key] = value
                        controlnet_canny.load_state_dict(new_state_dict)
                        controlnet_canny = controlnet_canny.to(device)

                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, controlnet_canny, control, unconditional_guidance_scale=scale,unconditional_conditioning=uc,)
                        
                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))
                                # output.save(os.path.join(outpath, content_name+'-'+prompt+f'-{grid_count:04}.png'))
                match = re.search(r'(gs-\d+)', embed_dir)
                if match:
                    result = match.group(1)
                code = result + '-' + str(seed) + '-' + str(strength) + '-' + str(ddim_steps)
                output.save(os.path.join(outpath, content_name + "-" + code + '.jpg'))
                grid_count += 1

                toc = time.time()
    return output 

model.embedding_manager.load(embed_dir)
model.embedding_manager.to(device)
model = model.to(device)

if hasattr(model, "embedding_manager"):
    model.embedding_manager = model.embedding_manager.to(device)
if hasattr(model, "cond_stage_model"):
    model.cond_stage_model = model.cond_stage_model.to(device)
if hasattr(model.cond_stage_model, "transformer"):
    model.cond_stage_model.transformer = model.cond_stage_model.transformer.to(device)
if hasattr(model.cond_stage_model, "text_model"):
    model.cond_stage_model.text_model = model.cond_stage_model.text_model.to(device)
if hasattr(model.cond_stage_model, "embeddings"):
    model.cond_stage_model.embeddings = model.cond_stage_model.embeddings.to(device)
if hasattr(model.cond_stage_model, "token_embedding"):
    model.cond_stage_model.token_embedding = model.cond_stage_model.token_embedding.to(device)

# main(prompt = '*', content_dir = contentdir, style_dir = contentdir, ddim_steps = 50, strength = 0.7, seed=42, model = model)
main(prompt = '*', \
    content_dir = contentdir, \
    ddim_steps = ddim_steps, \
    strength = strength, \
    seed=seed, \
    height = 512, \
    width = 768, \
    prospect_words = ['*'],
    model = model,\
    n_samples=n_samples, \
    )
    

