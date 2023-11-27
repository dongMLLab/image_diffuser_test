import time
from diffusers import StableDiffusionPipeline
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline
# from diffusers.pipelines import dance_diffusion
# from diffusers.utils import make_image_grid
import torch

def text_to_image(model: str, prompt: str, fileName: str):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Device: {}".format(DEVICE))

    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16,revision="fp16",).to("mps")
    pipe.enable_attention_slicing()
    pipe.load_textual_inversion("models/charturnerv2.pt", token="charturnerv2")

    print("model: {}".format(model))

    result = pipe(prompt, num_inference_steps=20, guidance_scale=9).images

    print("Generate Finished: {}".format(fileName))

    for r in range(len(result)):
        result[r].save("results/"+fileName+r+".png")

def single_file(prompt:str, model: str, fileName: str, time: str):
    pipeline = StableDiffusionPipeline.from_single_file(
        model
        # "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
    ).to("mps")

    pipeline.enable_attention_slicing()
    pipeline.load_textual_inversion("models/21charturnerv2.pt", token="charturnerv2")

    print("model: {}".format(model))

    result = pipeline(prompt, num_inference_steps=40, guidance_scale=11).images

    print("Generate Finished: {}".format(fileName))

    for r in range(len(result)):
        result[r].save("results/"+fileName+str(r)+".png")
        print("File saveed to {}".format("results/"+time+fileName+str(r)+".png"))

def generate_video():
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("mps")
    pipe.enable_model_cpu_offload()
    # memory optimization
    pipe.enable_vae_slicing()

    prompt = "Darth Vader surfing a wave"
    video_frames = pipe(prompt, num_frames=64).frames
    video_path = export_to_video(video_frames)

def main( model: str, prompt: str, fileName: str,mode="image"):
    start = time.time()
    if mode == "image":
        text_to_image(model, prompt, fileName)

    if mode == "single":
        single_file(prompt, model, fileName, str(start))

    if mode =="video":
        generate_video()

# "models/Realistic_Vision_V5.1.ckpt"
# "models/jyzjk.safetensors"
main("models/jyzjk.safetensors", "masterpiece, Adobe Photo, korean, realistic, beautiful", "test", "single")