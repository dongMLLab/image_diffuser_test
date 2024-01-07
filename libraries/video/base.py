from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import torch

def generate_video():
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("mps")
    pipe.enable_model_cpu_offload()
    # memory optimization
    pipe.enable_vae_slicing()

    prompt = "Darth Vader surfing a wave"
    video_frames = pipe(prompt, num_frames=64).frames
    video_path = export_to_video(video_frames)