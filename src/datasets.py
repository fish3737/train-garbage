import gzip
import shutil
import struct
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype


MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

GARBAGE_URL = "https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com/MindStudio-pc/data_en.zip"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, dest)


def download_mnist(root: str) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        download_file(url, root_path / name)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file: {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


class MnistGenerator:
    def __init__(self, root: str, train: bool, max_samples: int = 0):
        root_path = Path(root)
        prefix = "train" if train else "t10k"
        self.images = _read_idx_images(root_path / f"{prefix}-images-idx3-ubyte.gz")
        self.labels = _read_idx_labels(root_path / f"{prefix}-labels-idx1-ubyte.gz")
        if max_samples and max_samples > 0:
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index][..., None]
        return image, int(self.labels[index])


def create_mnist_dataset(root: str, train: bool, batch_size: int, max_samples: int = 0) -> ds.Dataset:
    dataset = ds.GeneratorDataset(MnistGenerator(root, train, max_samples=max_samples), ["image", "label"], shuffle=train)
    image_ops = [
        vision.Resize((32, 32)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=[0.1307], std=[0.3081]),
        vision.HWC2CHW(),
    ]
    dataset = dataset.map(image_ops, input_columns="image")
    dataset = dataset.map(transforms.TypeCast(mstype.int32), input_columns="label")
    return dataset.batch(batch_size, drop_remainder=train)


def download_garbage_zip(zip_path: str) -> None:
    download_file(GARBAGE_URL, Path(zip_path))


def extract_archive(archive_path: str, out_dir: str) -> None:
    archive = Path(archive_path)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        return
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tar":
        with tarfile.open(archive) as tf:
            tf.extractall(out)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def _image_files(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in root.rglob("*"):
        if path.suffix.lower() in exts and path.is_file():
            yield path


def prepare_image_folder(raw_dir: str, prepared_dir: str, train_ratio: float = 0.8, seed: int = 42) -> Dict[str, int]:
    raw = Path(raw_dir)
    prepared = Path(prepared_dir)
    if (prepared / "train").exists() and (prepared / "val").exists():
        classes = sorted([p.name for p in (prepared / "train").iterdir() if p.is_dir()])
        return {name: idx for idx, name in enumerate(classes)}

    class_to_files: Dict[str, List[Path]] = {}
    for image in _image_files(raw):
        label = image.parent.name
        class_to_files.setdefault(label, []).append(image)

    if len(class_to_files) < 2:
        raise RuntimeError(f"Cannot infer class folders under {raw}. Please check dataset structure.")

    rng = np.random.default_rng(seed)
    class_names = sorted(class_to_files)
    for split in ["train", "val"]:
        for name in class_names:
            (prepared / split / name).mkdir(parents=True, exist_ok=True)

    for name in class_names:
        files = sorted(class_to_files[name])
        rng.shuffle(files)
        cut = max(1, int(len(files) * train_ratio))
        for split, subset in [("train", files[:cut]), ("val", files[cut:])]:
            for src in subset:
                dst = prepared / split / name / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

    return {name: idx for idx, name in enumerate(class_names)}


class ImageFolderGenerator:
    def __init__(self, root: str, class_to_idx: Dict[str, int], image_size: int = 32):
        self.samples: List[Tuple[Path, int]] = []
        self.image_size = image_size
        for class_name, label in class_to_idx.items():
            class_dir = Path(root) / class_name
            for image in _image_files(class_dir):
                self.samples.append((image, label))
        if not self.samples:
            raise RuntimeError(f"No image files found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("L").resize((self.image_size, self.image_size))
        return np.asarray(image, dtype=np.uint8)[..., None], int(label)


def create_garbage_dataset(root: str, class_to_idx: Dict[str, int], batch_size: int, train: bool, image_size: int = 32) -> ds.Dataset:
    dataset = ds.GeneratorDataset(
        ImageFolderGenerator(root, class_to_idx, image_size=image_size),
        ["image", "label"],
        shuffle=train,
    )
    image_ops = [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=[0.5], std=[0.5]),
        vision.HWC2CHW(),
    ]
    dataset = dataset.map(image_ops, input_columns="image")
    dataset = dataset.map(transforms.TypeCast(mstype.int32), input_columns="label")
    return dataset.batch(batch_size, drop_remainder=train)


class RGBImageFolderGenerator:
    def __init__(self, root: str, class_to_idx: Dict[str, int], image_size: int = 64, train: bool = False):
        self.samples: List[Tuple[Path, int]] = []
        self.image_size = image_size
        self.train = train
        for class_name, label in class_to_idx.items():
            class_dir = Path(root) / class_name
            for image in _image_files(class_dir):
                self.samples.append((image, label))
        if not self.samples:
            raise RuntimeError(f"No image files found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.train:
            image = image.resize((self.image_size + 8, self.image_size + 8))
            max_offset = 8
            left = np.random.randint(0, max_offset + 1)
            top = np.random.randint(0, max_offset + 1)
            image = image.crop((left, top, left + self.image_size, top + self.image_size))
            if np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image = image.resize((self.image_size, self.image_size))
        return np.asarray(image, dtype=np.uint8), int(label)


def create_rgb_garbage_dataset(root: str, class_to_idx: Dict[str, int], batch_size: int, train: bool, image_size: int = 64) -> ds.Dataset:
    dataset = ds.GeneratorDataset(
        RGBImageFolderGenerator(root, class_to_idx, image_size=image_size, train=train),
        ["image", "label"],
        shuffle=train,
    )
    image_ops = [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vision.HWC2CHW(),
    ]
    dataset = dataset.map(image_ops, input_columns="image")
    dataset = dataset.map(transforms.TypeCast(mstype.int32), input_columns="label")
    return dataset.batch(batch_size, drop_remainder=train)
