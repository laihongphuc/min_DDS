import torch

def sds(unet,
        scheduler,
        z, 
        query_embeddings, 
        guidance_scale,
        noise=None, 
        min_noise_level=50, 
        max_noise_level=950):
    """
    Compute Score Distillation for Image Editing
    Args:
    - z: latent to optimization
    - init_image: 
    - query_embeddings: embeddings of query prompt
    """
    # sample timestep t
    t = torch.randint(min_noise_level, max_noise_level, (1, )).item()
    if noise is None:
        noise = torch.randn_like(z)
    noise_latent = scheduler.add_noise(z, noise, torch.tensor([torch.tensor(t)]))
    latent_model_input = torch.cat([noise_latent] * 2)
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=query_embeddings).sample
    # do guidance
    uncond_noise_pred, text_noise_pred = noise_pred.chunk(2)
    noise_pred = uncond_noise_pred + guidance_scale * (text_noise_pred - uncond_noise_pred)
    grad = noise_pred - noise 
    return grad.detach()


def dds(unet,
        scheduler,
        z,
        init_latent,
        ref_embeddings,
        query_embeddings, 
        guidance_scale, 
        min_noise_level=50, 
        max_noise_level=950):
    """
    Compute Score Distillation for Image Editing
    Args:
    - z: latent to optimization
    - init_image: 
    - ref_embeddings: embeddings of ref prompt
    - query_embeddings: embeddings of query prompt
    """
    # sample timestep t
    t = torch.randint(min_noise_level, max_noise_level, (1, )).item()
    noise = torch.randn_like(z)
    sds_loss_real = sds(unet, scheduler, init_latent, ref_embeddings, guidance_scale, noise, min_noise_level, max_noise_level)
    sds_loss_optimize = sds(unet, scheduler, z, query_embeddings, guidance_scale, noise, min_noise_level, max_noise_level)
    grad = sds_loss_optimize - sds_loss_real
    return grad.detach()
