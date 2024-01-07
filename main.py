import time
from safetensors.torch import load
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
import os
from libraries.image.base import single_file, text_to_image
from libraries.image.lora import lcm_lora
from libraries.video.base import generate_video
# from diffusers.pipelines import dance_diffusion
# from diffusers.utils import make_image_grid

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

def main(prompt: str, fileName: str, mode="image"):
    start = time.time()

    if mode == "image" or mode == "single":
        models = os.listdir("models/image")

    else:
        models = os.listdir("models/video")

    print("Model Lists: {}".format(models))
    model = int(input("Select Model Number: "))

    if mode == "image":
        text_to_image( models[model - 1], prompt, fileName)

    if mode == "single":
        single_file(prompt, models[model - 1], fileName, str(start))

    if mode =="video":
        generate_video()
    
    if mode =="lora":
        lcm_lora(prompt)

# "models/Realistic_Vision_V5.1.ckpt"
# "models/jyzjk.safetensors"
main("masterpiece, Donald Trumph, Joe Viden", "test.png", "lora")