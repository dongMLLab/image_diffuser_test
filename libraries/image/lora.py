from diffusers import  LCMScheduler, AutoPipelineForText2Image
import torch

# https://huggingface.co/latent-consistency/lcm-lora-sdxl
def lcm_lora(prompt: str, fileName: str, step=20, guidance_scale=7):
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id="latent-consistency/lcm-lora-sdxl"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("mps")

        # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
    
    generator = torch.manual_seed(0)
    image = pipe(prompt=prompt,num_inference_steps=step, guidance_scale=guidance_scale, generator=generator).images[0]
    image.save("results/"+fileName)
