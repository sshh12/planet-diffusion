# Planet Diffusion

> Fine-tuning stable diffusion to generate planet/moon textures.

## Using Stable Diffusion v1

### Demos (v1)

Cherry-picked best-of-4. It tends to struggle with prompts involving oceans or continents as that's pretty overfit to Earth. Generally, this model is fairly overfit to existing objects in our solar system.

|                               Prompt                               |                                                   Texture                                                   |                                                   Render                                                    |
| :----------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
|   a red-ish moon full of large volcanos and craters. fictitious    | ![Texture1](https://github.com/sshh12/planet-diffusion/assets/6625384/53a5344c-677a-4e12-a797-7e4336137e17) | ![Render1](https://github.com/sshh12/planet-diffusion/assets/6625384/dc4168c5-7d32-407f-8960-d4fc1b743ab3)  |
|    a large gas giant with multi-color rainbow bands. fictitious    | ![Texture2](https://github.com/sshh12/planet-diffusion/assets/6625384/99497404-a5d4-4b43-b516-a63b67f281a2) | ![untitled](https://github.com/sshh12/planet-diffusion/assets/6625384/6800cd5d-65dc-4fc8-87b3-3ddb6d604907) |
| a dark grey cratered planet with large white icy poles. fictitious |  ![00066](https://github.com/sshh12/planet-diffusion/assets/6625384/b4a57ebd-782e-4fef-a61e-c15a6cb78de1)   | ![untitled](https://github.com/sshh12/planet-diffusion/assets/6625384/f263a245-8c77-4e09-83de-69e6a21d2660) |

### Training (v1)

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

### Inference (v1)

```python
!(python scripts/txt2img.py \
    --prompt 'your prompt here' \
    --outdir '../outputs/' \
    --H 512 --W 1024 \
    --n_samples 2 \
    --config '../v1/planet-diffusion.yaml' \
    --ckpt 'logs/2023-06-29T00-13-09_planet-diffusion/checkpoints/epoch=000249.ckpt')
```
