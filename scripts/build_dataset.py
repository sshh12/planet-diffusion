from PIL import Image
import functools
import os
import glob
import openai
import json
import random

CAPTION_PROMPT = """
Describe the celestial object as if it was a prompt for an image generation model for the surface texture. 

* Do not include the name of the celestial object itself. 
* Caption only, it should not be a command. At most 2 sentences. 
* Be scientific, clear/exact, and not artistic. Note colors and high level features.
* The descriptions should be specific, visual, and mostly geological. Use plain concise language.
* Include the phrase "A planet/moon/etc cylindrical projection texture map of" in each caption

Generate 10 captions.
"""


def _generate_captions(name):
    functions = [
        {
            "name": "write_captions",
            "description": "Write captions",
            "parameters": {
                "type": "object",
                "properties": {"captions": {"type": "array", "items": {"type": "string"}}},
                "required": ["captions"],
            },
        }
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "system",
                "content": CAPTION_PROMPT,
            },
            {"role": "user", "content": name},
        ],
        functions=functions,
        function_call={"name": "write_captions"},
    )
    args = json.loads(resp.choices[0]["message"]["function_call"]["arguments"])
    return args["captions"]


@functools.lru_cache(maxsize=1000)
def generate_captions(name, retries=3):
    try:
        return _generate_captions(name)
    except openai.error.OpenAIError as e:
        if retries > 0:
            return generate_captions(name, retries=retries - 1)
        else:
            raise e


def build_dataset():
    i = 0
    with open(f"dataset/train/textures/metadata.jsonl", "w") as metacsv:
        for fn in glob.glob("textures/*.txt"):
            print(fn)
            id_ = os.path.splitext(os.path.basename(fn))[0]
            with open(fn, "r") as f:
                lines = f.readlines()
                title = lines[0].strip()
                data_source = lines[2].strip()
                img = Image.open(f"textures/{id_}.png")
            try:
                caption = random.choice(generate_captions(title))
            except Exception as e:
                print(fn, title, e)
                continue
            caption += " " + data_source
            img.convert("RGB").save(f"dataset/train/textures/{i}.png")
            meta = dict(file_name=f"{i}.png", text=caption)
            metacsv.write(f"{json.dumps(meta)}\n")
            i += 1


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/train/textures", exist_ok=True)
    build_dataset()
