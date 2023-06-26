from PIL import Image
from io import BytesIO
import requests
import re
import os

BASE_URL = "https://planet-texture-maps.fandom.com"
START_URL = BASE_URL + "/wiki/Local_Sitemap"
OUTPUT_SHAPE = (2048, 1024)


def _get_textures(page_html):
    textures = []
    try:
        title = re.search(r'firstHeading">\s+([\w ]+) Texture Maps\s*<', page_html).group(1)
    except AttributeError:
        return textures
    for row in re.findall(r"<tr>[\s\S]+?<\/?tr>", page_html):
        cols = re.findall(r'<td[ \w"=]+?>[\s\S]+?<\/?td>', row)
        if len(cols) != 5 or "download" not in row:
            continue
        image_url = BASE_URL + re.search(r'a href="([^"]+?)"', cols[1]).group(1)

        img_resp = requests.get(image_url)
        img_bytes = BytesIO(img_resp.content)
        img = Image.open(img_bytes)
        img_shape = img.size
        if not img_shape[0] == img_shape[1] * 2:
            continue

        data_source = re.search(r"<p>([\s\S]+?)<\/p>", cols[2]).group(1).strip()
        creator = re.search(r"<p>([\s\S]+?)<\/p>", cols[3]).group(1).strip()
        textures.append((title, image_url, data_source, creator, img_shape[0], img_shape[1], img))
    return textures


def _get_recursive_urls(page_html):
    urls = re.findall(r'<a href="(/wiki/[\w:]+)"', page_html)
    return [BASE_URL + url for url in urls if "://" not in url]


def download_textures():
    urls = _get_recursive_urls(requests.get(START_URL).text)
    visited = set()
    while urls:
        cur_url = urls.pop()
        visited.add(cur_url)
        page_html = requests.get(cur_url).text
        for title, image_url, data_source, creator, w, h, img in _get_textures(page_html):
            print(title, image_url, data_source, creator)
            id_ = abs(hash(image_url))
            img.resize(OUTPUT_SHAPE).save(f"textures/{id_}.png")
            with open(f"textures/{id_}.txt", "w") as f:
                f.write(f"{title}\n{image_url}\n{data_source}\n{creator}\n{w}\n{h}\n")


if __name__ == "__main__":
    os.makedirs("textures", exist_ok=True)
    download_textures()
