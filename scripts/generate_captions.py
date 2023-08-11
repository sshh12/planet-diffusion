import openai
import json
import random
import argparse

SEEDS = [
    "A turquoise-hued gas giant, streaked with swirling white wisps of high-altitude clouds, wrapped in a thin, multicolored ring system.",
    "A barren desert planet, coated in fine rusty-red sand, pockmarked with deep, dark craters, and cloaked in a thin, hazy atmosphere.",
    "A small, icy moon, encased in a shell of pure white ice and dust, covered with intricate patterns of frosty fissures.",
    "A metallic asteroid, featuring a rugged, heavily cratered surface with a shiny, silver-grey coloration.",
    "A vibrant blue planet, teeming with lush, tropical forests and deep, azure oceans, surrounded by a thick, oxygen-rich atmosphere.",
    "A dwarf planet exhibiting a striking purple color, with a surface peppered with craters and towering ice formations.",
    "A dusty, barren moon, characterized by a dull, yellowish-brown surface, marked by long, winding canyons and cliffs.",
    "A vast gas giant, deep green in color with turbulent, swirling storms visible across its gaseous surface.",
    "A small, irregularly-shaped celestial body, with a surface resembling an amalgam of silver and gold minerals, sprinkled with tiny, bright craters.",
    "A ringed planet, displaying a pale lavender surface, adorned with wide, white bands and an intricate network of icy rings.",
    "A rocky planet with an intense volcanic activity, marked by dark, lava-filled cracks and bright, sulfuric deposits.",
    "A flat and smooth celestial body, exhibiting a milky-white surface, devoid of any visible craters or mountains.",
    "A frozen moon with a surface of glittering, blue ice, marked by enormous, jagged ice ridges and deep chasms.",
    "A gas giant with a swirling, multihued surface of pink, purple, and blue, encircled by a ring of sparkling, ice-covered debris.",
    "A dense, rocky planet, with a surface covered in a network of enormous canyons, deep valleys, and towering mountains.",
    "An unusually shaped asteroid, with a bright, metallic surface, dotted with small, dark craters.",
    "A celestial body exhibiting a reddish-brown color, with a surface covered in a dense array of rocky spires and deep craters.",
    "A small moon with a smooth, glassy surface, believed to be the result of a layer of water-ice that has melted and refrozen.",
    "A distant exoplanet, with a bright green, photosynthetic biosphere, set against deep, dark oceans, under a thin, bluish atmosphere.",
    "A gas dwarf, with swirling bands of pastel blues and pinks, encased in a faint, dusty ring system.",
    "A distant celestial object with an icy crust, displaying a light blue shade, covered with round pits and rugged terrains.",
    "A dwarf planet with a reddish surface, covered in massive ice formations",
    "A planet with distinctive ring system, and a pale, yellowish brown surface with a splotchy appearance due to the bands",
    "An irregularly shaped moon with a spongy-looking surface, characterized by numerous small and large craters",
    "A celestial body showcasing a dense array of craters, with one large, prominent crater dominating one hemisphere",
    "A distressed terrain with heavy cratering and scarps, finished in a blend of brownish-grey",
    "A vast, yellowish gas giant with noticeable bands running parallel, surrounded by light-reflecting rings",
    "A minor moon featuring a rugged, uneven surface with a fair amount of large craters",
    "A celestial body speckled with an array of multicolored deposits from volcanic activities, namely bright yellows and dark browns.",
    "A moon with a smooth orangish surface covered in liquid methane lakes and complex organic compounds",
    "A dark grey moon, heavily scarred with craters, characterized by a sizable hollow and striations spreading from the center.",
    "A slowly rotating celestial object exhibiting a pinkish-gray color, with a somewhat elongated, tuber-like shape.",
]

CAPTION_PROMPT = """
{examples}

Generate {n} more captions for moons/planets/asteroids like the examples above.
"""


def generate_captions(n):
    samples = random.sample(SEEDS, 10)
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
                "content": CAPTION_PROMPT.format(n=n, examples=["- " + v for v in samples]),
            },
        ],
        temperature=0.8,
        functions=functions,
        function_call={"name": "write_captions"},
    )
    args = json.loads(resp.choices[0]["message"]["function_call"]["arguments"])
    captions = args["captions"]
    captions = [c.replace("- ", "") for c in captions]
    return captions


def build_captions(captions_path, count, batch_size):
    batch_size = min(batch_size, count)

    captions = set()
    while len(captions) < count:
        batch = generate_captions(batch_size)
        captions.update(batch)
        print(batch, f"{len(captions)}/{count}")
    with open(captions_path, "w") as f:
        for caption in captions:
            f.write(caption + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_path", type=str)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    build_captions(args.captions_path, args.count, args.batch_size)
