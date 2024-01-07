from diffusers import StableDiffusionPipeline
import torch

def single_file(prompt:str, model: str, fileName: str, time: str):
    try:
        # MODEL_ID = 'stabilityai/stable-diffusion-2-1'
        MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
        adapter_id="latent-consistency/lcm-lora-sdxl"

        pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID).to("mps")

        # if model[-11] == "safetensors":
        #     print("Model: {}".format(model[-11]))
        #     model_weights = load(model)
        #     pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID, model_weights).to("mps")

        # else:
        #     pipeline = StableDiffusionPipeline.from_single_file(
        #         model
        #         # "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        #     ).to("mps")
        # pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

        pipeline.enable_attention_slicing()
        # pipeline.load_textual_inversion("models/21charturnerv2.pt", token="charturnerv2")
    
        # print("model: {}".format(model))

        result = pipeline(prompt, num_inference_steps=40, guidance_scale=9, width=640, height=640).images

        print("Generate Finished: {}".format(fileName))

        for r in range(len(result)):
            result[r].save("results/"+fileName)
            print("File saveed to {}".format("results/"+fileName))
    except Exception as e:
        print(e)

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
        result[r].save("results/"+fileName)