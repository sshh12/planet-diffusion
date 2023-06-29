# Planet Diffusion

> Fine tuning stable diffusion to generate planet/moon textures.

## Training

1. Generate a dataset with the scripts in the repo or use [sshh12/planet-textures](https://huggingface.co/datasets/sshh12/planet-textures)
2. Clone https://github.com/justinpinkney/stable-diffusion @ `f1293f9795fda211d7fffdb84cd308424c2a184b` and apply `stable-diffusion.patch`
3. Train the model. I used a `NVIDIA RTX A6000` on LambdaLabs.

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt")

!(python main.py \
    -t \
    --base ../planet-diffusion.yaml \
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

Feel free to contact me for the original weights.

## Inference

```python
!(python scripts/txt2img.py \
    --prompt 'your prompt here' \
    --outdir '../outputs/' \
    --H 512 --W 1024 \
    --n_samples 2 \
    --config '../planet-diffusion.yaml' \
    --ckpt 'logs/2023-06-29T00-13-09_planet-diffusion/checkpoints/epoch=000249.ckpt')
```
