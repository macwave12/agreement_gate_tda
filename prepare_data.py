"""Download the 16 evaluation sets into the layout `main.py` reads, and cut the paper's fixed subsets.

    python prepare_data.py --data /path/to/data_root [--datasets dtd mnist ...]

- DTD, Flowers, Pets, EuroSAT, Aircraft, Caltech101, Food101: the original archives, with CoOp's split files
  (https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) placed in each dataset folder.
- Cars, GTSRB, FER2013, PCam, SVHN, MNIST, Rendered SST2, ImageNet-R: the test shards of the CLIP_benchmark
  mirrors on Hugging Face (`clip-benchmark/wds_*`), unpacked into one folder per class.
- ImageNet-Sketch: the archive of its Hugging Face mirror.
- Subsets: `random.Random(0)` draws 10,000 test images of Food101 and 3,000 each of FER2013, PCam, SVHN, MNIST.

About 20 GB of downloads. A finished dataset is skipped, so the script can be rerun after a failure.
"""
import argparse
import json
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

OX = "https://thor.robots.ox.ac.uk"
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
# dataset: (folder, archive url, {path in the archive: path in the folder}, {file in the folder: Google Drive id})
ARCHIVES = {
    "dtd": ("dtd", f"{OX}/dtd/dtd-r1.0.1.tar.gz", {"dtd/images": "images"},
            {"split_zhou_DescribableTextures.json": "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"}),
    "flower102": ("oxford_flowers", f"{OX}/flowers/102/102flowers.tgz", {"jpg": "jpg"},
                  {"split_zhou_OxfordFlowers.json": "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT"}),
    "pets": ("oxford_pets", f"{OX}/pets/images.tar.gz", {"images": "images"},
             {"split_zhou_OxfordPets.json": "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs"}),
    "eurosat": ("eurosat", "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip", {"EuroSAT_RGB": "2750"},
                {"split_zhou_EuroSAT.json": "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o"}),
    "aircraft": ("fgvc_aircraft", "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
                 "fgvc-aircraft-2013b.tar.gz", {"fgvc-aircraft-2013b/data": "."}, {}),
    "caltech101": ("caltech-101", "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
                   {"caltech-101/101_ObjectCategories.tar.gz": "101_ObjectCategories"},
                   {"split_zhou_Caltech101.json": "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN"}),
    "food101": ("food-101", "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz", {"food-101/images": "images"},
                {"split_zhou_Food101.json": "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl"}),
    "imagenet_sketch": ("imagenet-sketch", "https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/"
                        "data/ImageNet-Sketch.zip", {"sketch": "images"}, {}),
}
# dataset: (CLIP_benchmark name, folder, image folder, test shards used, subset size or None)
WDS = {
    "cars": ("cars", "stanford-cars-wds", "images", 5, None), "gtsrb": ("gtsrb", "gtsrb", "images", 3, None),
    "fer2013": ("fer2013", "fer2013", "images", 4, 3000), "pcam": ("vtab-pcam", "pcam", "images", 2, 3000),
    "svhn": ("vtab-svhn", "svhn", "images", 2, 3000), "mnist": ("mnist", "mnist", "images", 2, 3000),
    "rendered_sst2": ("renderedsst2", "rendered_sst2", "images", 4, None),
    "imagenet_r": ("imagenet-r", "imagenet-rendition", "imagenet-r", 4, None),
}
WDS_URL = "https://huggingface.co/datasets/clip-benchmark/wds_{}/resolve/main/test/{}.tar"
SEED, FOOD101_TEST = 0, 10_000


AGENT = {"User-Agent": "Mozilla/5.0 (agreement-gate dataset downloader; python-urllib)"}


def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120)


def download(url, dest):
    if not dest.exists():
        print(f"  downloading {url}", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with open_url(url) as response, open(part, "wb") as f:
            shutil.copyfileobj(response, f, 1 << 20)
        part.rename(dest)
    return dest


def unpack(archive, folder, moves):
    """Extract `archive` and move each path of `moves` into the dataset folder ("." = the folder itself)."""
    stage = folder / "_unpack"
    shutil.unpack_archive(str(archive), stage)
    for src, dst in moves.items():
        if (stage / src).is_file():  # an archive inside the archive (Caltech101)
            unpack(stage / src, folder, {dst: dst})
            continue
        for item in ([stage / src] if dst != "." else list((stage / src).iterdir())):
            shutil.move(str(item), str(folder / (dst if dst != "." else item.name)))
    shutil.rmtree(stage, ignore_errors=True)


def food101_subset(folder):
    """Keep CoOp's full split as `.full.json`; the split in use holds a fixed random 10,000 of its test images."""
    full, split = folder / "split_zhou_Food101.full.json", folder / "split_zhou_Food101.json"
    if not full.exists():
        shutil.copyfile(split, full)
    splits = json.loads(full.read_text(encoding="utf-8"))
    splits["test"] = sorted(random.Random(SEED).sample(splits["test"], FOOD101_TEST), key=lambda e: e[0])
    split.write_text(json.dumps(splits), encoding="utf-8")


def fetch_wds(key, folder, image_dir, name, shards):
    """Webdataset shards -> `<image_dir>/<class>/<image>`; a sample is `<key>.cls` (class index) plus one image.
    Class folders are the zero-padded index; ImageNet-R uses its sorted wnids, as `imagenet_r_mask` expects."""
    from data.imagenet_variants import imagenet_r_wnids
    wnids = sorted(imagenet_r_wnids)
    for i in range(shards):
        print(f"  shard {i + 1}/{shards}", flush=True)
        pending = {}
        with open_url(WDS_URL.format(name, i)) as response, tarfile.open(fileobj=response, mode="r|*") as tar:
            for member in tar:
                stem, dot, ext = member.name.rpartition(".")
                if not member.isfile() or not dot:
                    continue
                entry = pending.setdefault(stem, {})
                if ext.lower() == "cls":
                    entry["cls"] = int(tar.extractfile(member).read().decode().strip())
                elif ext.lower() in ("webp", "jpg", "jpeg", "png"):
                    entry["image"], entry["ext"] = tar.extractfile(member).read(), ext.lower()
                if "cls" in entry and "image" in entry:
                    d = folder / image_dir / (wnids[entry["cls"]] if key == "imagenet_r" else f"{entry['cls']:04d}")
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{Path(stem).name}.{entry['ext']}").write_bytes(entry["image"])
                    del pending[stem]


def subset(folder, image_dir, size):
    """Keep a fixed random `size` of the images; the others move to `<folder>/_excluded/`."""
    root = folder / image_dir
    files = sorted(p.relative_to(root).as_posix() for p in root.glob("*/*") if p.is_file())
    keep = set(random.Random(SEED).sample(files, size))
    for rel in files:
        if rel not in keep:
            (folder / "_excluded" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / rel), str(folder / "_excluded" / rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="dataset root to fill")
    ap.add_argument("--datasets", nargs="+", default=[*ARCHIVES, *WDS], choices=[*ARCHIVES, *WDS])
    a = ap.parse_args()
    root = Path(a.data)
    for key in a.datasets:
        folder = root / (ARCHIVES[key][0] if key in ARCHIVES else WDS[key][1])
        done = folder / "_prepared"
        if done.exists():
            print(f"[{key}] already prepared")
            continue
        print(f"[{key}] -> {folder}", flush=True)
        folder.mkdir(parents=True, exist_ok=True)
        if key in ARCHIVES:
            _, url, moves, drive = ARCHIVES[key]
            unpack(download(url, root / "_archives" / url.rsplit("/", 1)[1]), folder, moves)
            for name, file_id in drive.items():
                download(DRIVE.format(file_id), folder / name)
            if key == "food101":
                food101_subset(folder)
        else:
            name, _, image_dir, shards, size = WDS[key]
            fetch_wds(key, folder, image_dir, name, shards)
            if size:
                subset(folder, image_dir, size)
        done.write_text("ok", encoding="utf-8")
    print("done; the downloaded archives are in", root / "_archives", "and can be deleted")


if __name__ == "__main__":
    main()
