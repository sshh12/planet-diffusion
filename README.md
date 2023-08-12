# Planet Diffusion

> Fine-tuning stable diffusion to generate planet/moon textures.

## Using Stable Diffusion XL + LoRA [v2]

### Demos

Cherry-picked best of several generations with varying checkpoints, guidance scales, and seeds.

|                                                             Prompt                                                              |                                                                                                                    Texture                                                                                                                    |                                                                                                                        Render                                                                                                                        |
| :-----------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   A planet with a surface of turquoise and gold, marked by large, glittering rivers of molten metal and bright, shining seas.   |  ![00001_a_planet_with_a_surface_of_turquoise_and_gold_marked_by_large_glittering_rivers_of_molten_metal_and_bright_shining_seas_1300_15_0](https://github.com/sshh12/planet-diffusion/assets/6625384/e6b122df-f433-415a-a5b5-c2915e32210a)   |  ![00001_a_planet_with_a_surface_of_turquoise_and_gold_marked_by_large_glittering_rivers_of_molten_metal_and_bright_shining_seas_1300_15_0-render](https://github.com/sshh12/planet-diffusion/assets/6625384/56f36693-5e73-4301-95d9-b9bca233c1d2)   |
| A small, rocky planet with a sandy, yellow surface, characterized by its large, winding canyons and massive, towering volcanoes | ![00023_a_small_rocky_planet_with_a_sandy_yellow_surface_characterized_by_its_large_winding_canyons_and_massive_towering_volcanoes_1300_20_0](https://github.com/sshh12/planet-diffusion/assets/6625384/36fb05c3-de4b-4e79-885a-ac9162e7de30) | ![00023_a_small_rocky_planet_with_a_sandy_yellow_surface_characterized_by_its_large_winding_canyons_and_massive_towering_volcanoes_1300_20_0-render](https://github.com/sshh12/planet-diffusion/assets/6625384/f66e329d-129b-4d53-a811-634e4fe27c66) |
|   A planet with a surface covered in lush, green vegetation and large bodies of water, with a strikingly colorful atmosphere.   |  ![00036_a_planet_with_a_surface_covered_in_lush_green_vegetation_and_large_bodies_of_water_with_a_strikingly_colorful_atmosphere_2000_15_0](https://github.com/sshh12/planet-diffusion/assets/6625384/84c526ad-4324-4536-a5a7-e635b5b047a4)  |  ![00036_a_planet_with_a_surface_covered_in_lush_green_vegetation_and_large_bodies_of_water_with_a_strikingly_colorful_atmosphere_2000_15_0-render](https://github.com/sshh12/planet-diffusion/assets/6625384/c816272a-ae5e-44d9-aae9-d6fd94ee3f7d)  |
|            A frozen moon with a pristine white surface adorned with deep blue ice crevasses and towering ice peaks.             |         ![00055_a_frozen_moon_with_a_pristine_white_surface_adorned_with_deep_blue_ice_crevasses_and_towering_ice_peaks_2000_15_100](https://github.com/sshh12/planet-diffusion/assets/6625384/1045cb36-e6c2-407e-8dfe-ccacda92c5ac)          |         ![00055_a_frozen_moon_with_a_pristine_white_surface_adorned_with_deep_blue_ice_crevasses_and_towering_ice_peaks_2000_15_100-render](https://github.com/sshh12/planet-diffusion/assets/6625384/acae483b-11de-4fe0-8166-1b1839faa7bf)          |
| A dense, greenish gas giant, surrounded by a hazy, nebulous atmosphere, with faint, swirling bands visible across its surface.  |  ![00053_a_dense_greenish_gas_giant_surrounded_by_a_hazy_nebulous_atmosphere_with_faint_swirling_bands_visible_across_its_surface_1500_15_0](https://github.com/sshh12/planet-diffusion/assets/6625384/0770cb6e-9d60-4761-9ec8-4d4dcba225ec)  |  ![00053_a_dense_greenish_gas_giant_surrounded_by_a_hazy_nebulous_atmosphere_with_faint_swirling_bands_visible_across_its_surface_1500_15_0-render](https://github.com/sshh12/planet-diffusion/assets/6625384/d7ab70a2-9d55-46e5-9070-45b46b11037e)  |
|          A gas giant with a turbulent, stormy surface, displaying a mesmerizing swirl of teal, navy, and violet hues.           |          ![00047_a_gas_giant_with_a_turbulent_stormy_surface_displaying_a_mesmerizing_swirl_of_teal_navy_and_violet_hues_1300_15_0](https://github.com/sshh12/planet-diffusion/assets/6625384/04ac655e-22b3-4d4d-b80a-81dea4c7aed4)           |          ![00047_a_gas_giant_with_a_turbulent_stormy_surface_displaying_a_mesmerizing_swirl_of_teal_navy_and_violet_hues_1300_15_0-render](https://github.com/sshh12/planet-diffusion/assets/6625384/510849cb-24c5-4d25-ae99-6fb26d4eea02)           |

### Training

Follow the instructions on [sdxl-lora-planet-textures](https://huggingface.co/sshh12/sdxl-lora-planet-textures). You can also find several pre-trained models here.

LoRA enabled training on an `NVIDIA 3090 Ti`.

### Inference

```py
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.load_lora_weights("sshh12/sdxl-lora-planet-textures")
pipe.to("cuda")

prompt = "A dwarf planet exhibiting a striking purple color, with a surface peppered with craters and towering ice formations"
negative_prompt = 'blurry, fuzzy, low resolution, cartoon, painting'

image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=1024, height=512).images[0]
image
```

See `scripts/generate_images.py` for an example of advanced usage (including using an upscaler).

## Using Stable Diffusion [v1]

### Demos

Cherry-picked best-of-4. It tends to struggle with prompts involving oceans or continents as that's pretty overfit to Earth. Generally, this model is fairly overfit to existing objects in our solar system.

|                               Prompt                               |                                                   Texture                                                   |                                                   Render                                                    |
| :----------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
|   a red-ish moon full of large volcanos and craters. fictitious    | ![Texture1](https://github.com/sshh12/planet-diffusion/assets/6625384/53a5344c-677a-4e12-a797-7e4336137e17) | ![Render1](https://github.com/sshh12/planet-diffusion/assets/6625384/dc4168c5-7d32-407f-8960-d4fc1b743ab3)  |
|    a large gas giant with multi-color rainbow bands. fictitious    | ![Texture2](https://github.com/sshh12/planet-diffusion/assets/6625384/99497404-a5d4-4b43-b516-a63b67f281a2) | ![untitled](https://github.com/sshh12/planet-diffusion/assets/6625384/6800cd5d-65dc-4fc8-87b3-3ddb6d604907) |
| a dark grey cratered planet with large white icy poles. fictitious |  ![00066](https://github.com/sshh12/planet-diffusion/assets/6625384/b4a57ebd-782e-4fef-a61e-c15a6cb78de1)   | ![untitled](https://github.com/sshh12/planet-diffusion/assets/6625384/f263a245-8c77-4e09-83de-69e6a21d2660) |

### Training

1. Generate a dataset with the scripts in the repo or use [sshh12/planet-textures](https://huggingface.co/datasets/sshh12/planet-textures)
2. Clone https://github.com/justinpinkney/stable-diffusion @ `f1293f9795fda211d7fffdb84cd308424c2a184b` and apply `v1/stable-diffusion.patch`
3. Train the model. I used a `NVIDIA RTX A6000` on LambdaLabs. If you do everything correctly the first time, the expected cost is $12.

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt")

!(python main.py \
    -t \
    --base ../v1/planet-diffusion.yaml \
    --gpus "1" \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from "$ckpt_path" \
    data.params.batch_size=1 \
    lightning.trainer.accumulate_grad_batches=8 \
    data.params.validation.params.n_gpus=1 \
)
```

Feel free to contact me (using GitHub issues) for the original weights or you run into issues setting this up.

### Inference

```python
!(python scripts/txt2img.py \
    --prompt 'your prompt here' \
    --outdir '../outputs/' \
    --H 512 --W 1024 \
    --n_samples 2 \
    --config '../v1/planet-diffusion.yaml' \
    --ckpt 'logs/2023-06-29T00-13-09_planet-diffusion/checkpoints/epoch=000249.ckpt')
```
