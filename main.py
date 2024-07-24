from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from diffusers import DPMSolverMultistepScheduler
import numpy as np 
import matplotlib.pyplot as plt

from loss import sds, dds


# setup
model_id = "stabilityai/stable-diffusion-2-1-base"

# hyperparameter
magic_parameters = 2000             # i don't know why???  
lr = 1e-1                           # learning rate
guidance_scale = 7.5                # guidance scale for classifier-free guidance
min_noise_level = 50                # min noise level to do optimization 
max_noise_level = 950               # max noise level to do optimization
num_iters = 500                     # numbers of iterations

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

device = "cuda"

vae.to(device)
text_encoder.to(device)
unet.to(device)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# real image and prompt
image_path = "./content/flamingo.png"
init_image = Image.open(image_path).resize((512, 512))
ref_prompt=["a flamingo rollerskating"]
query_prompt = ["a stork rollerskating"]

# vae
def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# text embeddings
ref_input = tokenizer(ref_prompt, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
ref_embeddings = text_encoder(ref_input.input_ids.to(device))[0]
query_input = tokenizer(query_prompt, max_length=tokenizer.model_max_length, truncation=True, padding="max_length", return_tensors="pt")
query_embeddings = text_encoder(query_input.input_ids.to(device))[0]
# uncond embedding
uncond_input = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

ref_embeddings = torch.cat([uncond_embeddings, ref_embeddings], dim=0)
query_embeddings = torch.cat([uncond_embeddings, query_embeddings], dim=0)


def latent_optimization(init_image, 
                        ref_embeddings, 
                        query_embeddings,
                        guidance_scale, 
                        min_noise_level=50, 
                        max_noise_level=950, 
                        num_iters=200):
    init_latent = pil_to_latent(init_image)
    z = init_latent.clone()
    z.requires_grad = True
    optimizer = torch.optim.SGD([z], lr=1e-1)
    for i in range(num_iters):
        grad = dds(unet, scheduler, z, init_latent, ref_embeddings, query_embeddings, guidance_scale, min_noise_level, max_noise_level)
        # stop gradient to prevent computing Jacobian 
        loss_z = z * grad.detach()
        loss_z = loss_z.sum() / (z.shape[2] * z.shape[3])
        # ????????????????????????????
        (magic_parameters * loss_z).backward()
        print(f"Iter {i} loss {2000 * loss_z.item()}")
        optimizer.step()
        optimizer.zero_grad()
        if i % 20 == 0:
            image = latents_to_pil(z)[0]
            plt.imshow(np.array(image))
            plt.show()
    return z

if __name__ == "__main__":
    result = latent_optimization(init_image, ref_embeddings, query_embeddings, num_iters=500)