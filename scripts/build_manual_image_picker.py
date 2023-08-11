import torch
import os
import argparse
import tqdm
import json
from diffusers import DiffusionPipeline, AutoencoderKL


LORA_VALUE = "LORA_VALUE"
NEGATIVE_PROMPT = "blurry, fuzzy, low resolution, cartoon, painting"
NUM_INFERENCE_STEPS = 100


def build_manual_image_picker(
    captions_path, data_path, vae, model, cache_dir, lora, lora_values, gs_values, seed_values
):
    with open(captions_path, "r") as f:
        captions = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    vae_kl = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16, cache_dir=cache_dir)
    pipe = DiffusionPipeline.from_pretrained(
        model,
        vae=vae_kl,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    pipe.to("cuda")

    for idx, caption in tqdm.tqdm(enumerate(captions), total=len(captions)):
        path = os.path.join(
            data_path, f"{idx:05d}_" + caption.replace(" ", "_").replace(",", "").replace(".", "").lower()
        )
        os.makedirs(path, exist_ok=True)

        pipe_args = dict(
            prompt=caption,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            width=1024,
            height=512,
        )
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(
                dict(
                    model=model,
                    vae=vae,
                    lora=lora,
                    **pipe_args,
                ),
                f,
            )

        for lora_val in lora_values.split(","):
            lora_path = lora.replace(LORA_VALUE, lora_val)
            pipe.load_lora_weights(lora_path, cache_dir=cache_dir)
            for gs in gs_values.split(","):
                for seed in seed_values.split(","):
                    generator = torch.Generator(device="cuda").manual_seed(int(seed))
                    image = pipe(
                        **pipe_args,
                        generator=generator,
                        guidance_scale=int(gs),
                    ).images[0]
                    image.save(os.path.join(path, f"img_{lora_val}_{gs}_{seed}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--vae", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--lora", type=str, default="sshh12/sdxl-lora-planet-textures")
    parser.add_argument("--lora_values", type=str, default="1300,1500,2000,2200")
    parser.add_argument("--gs_values", type=str, default="10,15,20")
    parser.add_argument("--seed_values", type=str, default="0,50,100")

    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    build_manual_image_picker(
        args.captions_path,
        args.data_path,
        args.vae,
        args.model,
        args.cache_dir,
        args.lora,
        args.lora_values,
        args.gs_values,
        args.seed_values,
    )
