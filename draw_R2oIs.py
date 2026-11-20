"""Draw PCED multi-scale relation-evidence heatmaps.

The script has two input modes.  ``--input_mode single`` reads one image from
``--image_path`` and draws its four maps.  ``--input_mode dataset`` (the
default when no image path is supplied) randomly scans the test set and draws
one correct and one incorrect prediction for every PISC-Coarse/Fine class.
For PIPA, only the requested small number of correct and incorrect examples
is selected.  Each result is saved as
``relation_label/{correct,incorrect}_pred_{label}/image_name/{four PNG files}``.

The dataset mode follows the 11-field item returned by ``train_vit(4).py``.
For a standalone image, the image is reused as the full image, union crop, and
both person crops, while masks and spatial configuration are set to zero.
Each relation produces 1x1, 2x2, 4x4, and an overlap-aware union map.  Nearby
heat regions are connected with a soft spatial tolerance before fusion.  A
pixel covered by one scale keeps that scale's response; a pixel covered by
multiple nearby scales averages only those active responses.  The selected
window sizes are 16, 32, and 64 pixels for the default 224x224 input.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms


INPUT_SIZE = 224
BASE_GRID = 14
PATCH_SIZE = INPUT_SIZE // BASE_GRID
SCALES = (1, 2, 4)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
MODE_NUM_CLASSES = {"PISC_Coarse": 3, "PISC_Fine": 6, "PIPA": 16}
SCALE_COLORS = ("#00D7FF", "#FFD166", "#EF476F")
DEFAULT_DATA_ROOT = {
    "PISC_Coarse": "xxx/inputFile/image",
    "PISC_Fine": "xxx/inputFile/image",
    "PIPA": "xxx/inputFile/pipa",
}
DEFAULT_ANNOTATION_FILE = {
    "PISC_Coarse": "xxx/PISCF_test.pkl",
    "PISC_Fine": "xxx/PISCC_test.pkl",
    "PIPA": "xxx/PIPA_test.pkl",
}


def resolve_path(value: str, anchor: Optional[Path] = None) -> Path:
    """Resolve a user path without changing an explicit absolute path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    if anchor is not None:
        candidates.append(anchor / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def import_from_path(path: str, module_name: str):
    module_path = resolve_path(path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Python module does not exist: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python file: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load common checkpoint layouts and remove DataParallel prefixes."""
    payload = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    if isinstance(payload, torch.nn.Module):
        payload = payload.state_dict()
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping):
        raise TypeError("The checkpoint must contain a state_dict dictionary.")

    state: Dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "network.", "net.")
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        state[key] = value
    if not state:
        raise ValueError(f"No tensor parameters were found in {checkpoint_path}.")
    return state


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def as_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return as_name(value[0])
    return str(value)


def as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def image_batch(value: Any) -> torch.Tensor:
    if isinstance(value, Image.Image):
        value = make_transform()(value.convert("RGB"))
    tensor = as_tensor(value, torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(
            f"Expected image tensor [R,3,H,W], received {tuple(tensor.shape)}."
        )
    return tensor


def full_image_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, Image.Image):
        value = make_transform()(value.convert("RGB"))
    tensor = as_tensor(value, torch.float32)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"Expected full image tensor [3,H,W], received {tuple(tensor.shape)}."
        )
    return tensor


def mask_batch(value: Any) -> torch.Tensor:
    tensor = as_tensor(value, torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0 if tensor.shape[0] == 1 else 1)
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected mask tensor [R,1,H,W], received {tuple(tensor.shape)}."
        )
    if tensor.shape[1] != 1:
        tensor = tensor[:, :1]
    return tensor


def relation_batch(value: Any, width: int, dtype: torch.dtype) -> torch.Tensor:
    tensor = as_tensor(value, dtype)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(
            f"Expected relation tensor [R,{width}], received {tuple(tensor.shape)}."
        )
    return tensor


def target_batch(value: Any) -> torch.Tensor:
    return as_tensor(value, torch.long).reshape(-1)


def count_batch(value: Any) -> torch.Tensor:
    return as_tensor(value, torch.long).reshape(-1)


def collate_relations(data: Sequence[Tuple[Any, ...]]):
    """Collate the same 11 fields returned by train_vit(4).py."""
    names: List[str] = []
    full_images: List[torch.Tensor] = []
    unions: List[torch.Tensor] = []
    person1: List[torch.Tensor] = []
    person2: List[torch.Tensor] = []
    pair_masks: List[torch.Tensor] = []
    interaction_masks: List[torch.Tensor] = []
    spatial: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    pair_indices: List[torch.Tensor] = []
    relation_counts: List[torch.Tensor] = []

    for item in data:
        if len(item) != 11:
            raise ValueError(
                "Each sample must contain 11 fields: name, full image, three "
                "crops, two masks, spatial config, target, pair index, and count."
            )
        (
            name,
            full_image,
            union,
            obj1,
            obj2,
            pair_mask,
            interaction_mask,
            bpos,
            target,
            pair_index,
            relation_count,
        ) = item
        names.append(as_name(name))
        full_images.append(full_image_tensor(full_image))
        unions.append(image_batch(union))
        person1.append(image_batch(obj1))
        person2.append(image_batch(obj2))
        pair_masks.append(mask_batch(pair_mask))
        interaction_masks.append(mask_batch(interaction_mask))
        spatial.append(relation_batch(bpos, 6, torch.float32))
        targets.append(target_batch(target))
        pair_indices.append(relation_batch(pair_index, 2, torch.long))
        relation_counts.append(count_batch(relation_count))

    counts = torch.cat(relation_counts, dim=0)
    relation_count = int(counts.sum().item())
    if relation_count != sum(t.shape[0] for t in unions):
        raise ValueError("Image relation counts do not match the number of crops.")
    return (
        names,
        torch.stack(full_images, dim=0),
        torch.cat(unions, dim=0),
        torch.cat(person1, dim=0),
        torch.cat(person2, dim=0),
        torch.cat(pair_masks, dim=0),
        torch.cat(interaction_masks, dim=0),
        torch.cat(spatial, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(pair_indices, dim=0),
        counts,
    )


def resolve_relative(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_image_tensor(path: Path, transform: transforms.Compose) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def load_mask(value: Any, base_dir: Path) -> torch.Tensor:
    """Read an optional mask and return [1,224,224]; missing means all zeros."""
    if value is None:
        return torch.zeros(1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    if isinstance(value, (list, tuple)):
        tensor = as_tensor(value, torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("Inline masks must have shape [H,W] or [1,H,W].")
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]
    path = resolve_relative(value, base_dir)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Numpy mask must be 2-D: {path}")
        tensor = tensor.unsqueeze(0)
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class ManifestRelationDataset(Dataset):
    """Read one relation per JSON record."""

    def __init__(self, manifest_path: str, default_label: int = -1):
        self.manifest_path = resolve_path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if isinstance(records, dict):
            records = records.get("samples", records.get("records", records))
        if not isinstance(records, list):
            raise ValueError("The manifest must be a JSON list of relation records.")
        self.records = records
        self.base_dir = self.manifest_path.parent
        self.transform = make_transform()
        self.default_label = int(default_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        full_value = record.get("full_image", record.get("image"))
        required = ("union_crop", "person1_crop", "person2_crop")
        if full_value is None or any(key not in record for key in required):
            raise KeyError(
                "A manifest record requires full_image/image, union_crop, "
                "person1_crop, and person2_crop."
            )
        full = load_image_tensor(resolve_relative(full_value, self.base_dir), self.transform)
        union = load_image_tensor(
            resolve_relative(record["union_crop"], self.base_dir), self.transform
        )
        obj1 = load_image_tensor(
            resolve_relative(record["person1_crop"], self.base_dir), self.transform
        )
        obj2 = load_image_tensor(
            resolve_relative(record["person2_crop"], self.base_dir), self.transform
        )
        pair_mask = load_mask(record.get("pair_mask"), self.base_dir).unsqueeze(0)
        interaction_mask = load_mask(
            record.get("interaction_mask"), self.base_dir
        ).unsqueeze(0)
        spatial = as_tensor(
            record.get("spatial_config", record.get("bpos", [0.0] * 6)),
            torch.float32,
        ).reshape(1, 6)
        label = int(record.get("label", record.get("target", self.default_label)))
        pair_index = as_tensor(
            record.get("pair_indices", [[0, 1]]), torch.long
        ).reshape(1, 2)
        name = record.get("name", Path(str(full_value)).stem)
        return (
            name,
            full,
            union.unsqueeze(0),
            obj1.unsqueeze(0),
            obj2.unsqueeze(0),
            pair_mask,
            interaction_mask,
            spatial,
            torch.tensor([label], dtype=torch.long),
            pair_index,
            torch.tensor([1], dtype=torch.long),
        )


def expand_image_inputs(
    image_paths: Sequence[str],
    image_dirs: Sequence[str],
    image_list: Optional[str],
) -> List[Path]:
    raw = list(image_paths)
    raw.extend(image_dirs)
    if image_list:
        list_path = resolve_path(image_list)
        with list_path.open("r", encoding="utf-8") as handle:
            raw.extend(line.strip() for line in handle if line.strip())

    paths: List[Path] = []
    for item in raw:
        item = str(item)
        path = Path(item).expanduser()
        if any(char in item for char in "*?["):
            candidates = [Path(candidate) for candidate in glob.glob(item, recursive=True)]
        elif path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            candidates = [path]
        paths.extend(candidate for candidate in candidates if candidate.is_file())

    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    if not unique:
        raise ValueError("No input images were found.")
    return unique


class ImageOnlyRelationDataset(Dataset):
    """Use one image as all visual inputs for quick standalone visualization."""

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
    ):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = make_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = load_image_tensor(path, self.transform)
        zero_mask = torch.zeros(1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
        return (
            str(path),
            image,
            image.unsqueeze(0),
            image.unsqueeze(0),
            image.unsqueeze(0),
            zero_mask.unsqueeze(0),
            zero_mask.unsqueeze(0),
            torch.zeros(1, 6, dtype=torch.float32),
            torch.tensor([self.labels[index]], dtype=torch.long),
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )


def read_label_list(path: Optional[str], expected: int, default_label: int) -> List[int]:
    if path is None:
        return [int(default_label)] * expected
    label_path = resolve_path(path)
    labels = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                labels.append(int(line.split(",")[0]))
    if len(labels) != expected:
        raise ValueError(
            f"{label_path} contains {len(labels)} labels, but {expected} images were found."
        )
    return labels


def build_dataset(args) -> Dataset:
    if args.input_mode == "single" or args.image_path:
        if not args.image_path:
            raise ValueError("--image_path is required in single-image mode.")
        return ImageOnlyRelationDataset(
            [resolve_path(args.image_path)],
            [int(args.single_label)],
        )

    if args.input_mode not in ("auto", "dataset"):
        raise ValueError(
            "No input was specified. Use --image_path for one image or "
            "--input_mode dataset for the representative test-set mode."
        )

    dataset_root = args.dataset_root or DEFAULT_DATA_ROOT[args.mode]
    annotation_file = args.annotation_file or DEFAULT_ANNOTATION_FILE[args.mode]
    if dataset_root and annotation_file:
        module = import_from_path(args.dataset_module, "dataset_module_heatmaps")
        if not hasattr(module, "DataSet"):
            raise AttributeError(f"{args.dataset_module} does not define DataSet.")
        transform = make_transform()
        return module.DataSet(
            dataset_root,
            annotation_file,
            transform,
            transform,
        )


def load_model(args, device: torch.device):
    module = import_from_path(args.network_file, "network_module_heatmaps")
    vit_checkpoint = args.vit_checkpoint
    if vit_checkpoint is None:
        vit_checkpoint = getattr(module, "DEFAULT_VIT_CHECKPOINT", None)
    if vit_checkpoint is None:
        raise ValueError("Pass --vit_checkpoint for this network.")

    model = module.network(
        num_classes=args.num_classes,
        vit_checkpoint=vit_checkpoint,
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        freeze_backbone=True,
    )
    state = load_state_dict(args.checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warning] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[warning] Unexpected checkpoint keys: {len(unexpected)}")
    model.to(device).eval()
    return model


@torch.inference_mode()
def forward_with_raw_attention(model, batch, device: torch.device):
    (
        _names,
        full_image,
        union,
        obj1,
        obj2,
        pair_mask,
        interaction_mask,
        spatial_config,
        _target,
        _pair_indices,
        relation_counts,
    ) = batch
    full_image = full_image.to(device, non_blocking=True)
    union = union.to(device, non_blocking=True)
    obj1 = obj1.to(device, non_blocking=True)
    obj2 = obj2.to(device, non_blocking=True)
    pair_mask = pair_mask.to(device, non_blocking=True)
    interaction_mask = interaction_mask.to(device, non_blocking=True)
    spatial_config = spatial_config.to(device, non_blocking=True)
    relation_counts = relation_counts.to(device, non_blocking=True)

    union_feature, spatial_feature, pair_query = model._encode_pair_features(
        union, obj1, obj2, spatial_config
    )
    _, full_patch_tokens = model.visual_encoder(
        full_image, return_patch_tokens=True
    )
    full_patch_tokens = model._expand_full_image_features(
        full_patch_tokens,
        image_relation_counts=relation_counts,
        relation_count=union.shape[0],
    )
    spatial_token_features = model._build_spatial_token_features(
        pair_mask, interaction_mask
    )
    multi_scale_tokens = model.multi_scale_encoder(full_patch_tokens)
    attended_feature, token_attention = model.cross_attention(
        pair_query,
        multi_scale_tokens,
        spatial_token_features,
    )
    classifier_input = torch.cat(
        [attended_feature, pair_query, union_feature, spatial_feature], dim=-1
    )
    logits = model.classifier(classifier_input)
    return logits, token_attention.detach().cpu()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        return np.ones_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def project_topk_heatmap(
    token_scores: np.ndarray,
    scale: int,
    top_k: int,
    sigma_factor: float,
) -> np.ndarray:
    """Render top-k tokens as continuous Gaussian evidence fields.

    The Gaussian centre is the centre of the corresponding sliding window and
    its standard deviation grows with the receptive-field width.  Thus a 1x1
    token produces a small smooth hotspot, while a 4x4 token produces a broad
    smooth region.  The returned map keeps the original attention magnitudes;
    normalization is applied only when a map is rendered.
    """
    rows = BASE_GRID - scale + 1
    scores = np.asarray(token_scores, dtype=np.float32).reshape(-1)
    expected = rows * rows
    if scores.size != expected:
        raise ValueError(f"Scale {scale} expects {expected} tokens, got {scores.size}.")
    count = max(1, min(int(top_k), scores.size))
    selected = np.argsort(scores)[-count:][::-1]
    selected_values = np.maximum(scores[selected], 0.0)
    if float(selected_values.max()) <= 0:
        selected_values = normalize(scores[selected])

    yy, xx = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
    heat = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    for token_index, value in zip(selected, selected_values):
        row, column = divmod(int(token_index), rows)
        center_x = (column + 0.5 * scale) * PATCH_SIZE
        center_y = (row + 0.5 * scale) * PATCH_SIZE
        sigma = max(1.0, float(sigma_factor) * scale * PATCH_SIZE)
        gaussian = np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2)
            / (2.0 * sigma * sigma)
        )
        heat += float(value) * gaussian.astype(np.float32)
    return heat


def normalize_heatmap(heat: np.ndarray, gamma: float) -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    heat = heat - float(heat.min())
    maximum = float(heat.max())
    if maximum > 1e-8:
        heat = heat / maximum
    return np.power(np.clip(heat, 0.0, 1.0), max(float(gamma), 1e-3))


def resize_heatmap(heat: np.ndarray, width: int, height: int) -> np.ndarray:
    heat_image = Image.fromarray(
        (np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    resampling = getattr(Image, "Resampling", Image)
    heat_image = heat_image.resize((width, height), resampling.BILINEAR)
    return np.asarray(heat_image, dtype=np.float32) / 255.0


def _soft_support_mask(
    heat: np.ndarray,
    threshold: float,
    transition: float,
    overlap_radius: int,
    blur_radius: float,
) -> np.ndarray:
    """Build a soft, spatially tolerant support mask for one scale."""
    heat = np.asarray(heat, dtype=np.float32)
    transition = max(float(transition), 1e-4)
    lower = max(float(threshold) - transition, 0.0)
    upper = min(float(threshold) + transition, 1.0)
    support = np.clip((heat - lower) / max(upper - lower, 1e-4), 0.0, 1.0)
    support = support * support * (3.0 - 2.0 * support)

    support_image = Image.fromarray(
        (support * 255.0).astype(np.uint8),
        mode="L",
    )
    radius = max(int(overlap_radius), 0)
    if radius > 0:
        support_image = support_image.filter(
            ImageFilter.MaxFilter(size=2 * radius + 1)
        )
    if float(blur_radius) > 0:
        support_image = support_image.filter(
            ImageFilter.GaussianBlur(radius=float(blur_radius))
        )
    return np.asarray(support_image, dtype=np.float32) / 255.0


def fuse_overlap_aware(
    scale_maps: Sequence[np.ndarray],
    threshold: float,
    transition: float,
    overlap_radius: int,
    blur_radius: float,
) -> np.ndarray:
    """Fuse scale maps with soft region overlap instead of hard pixel cuts.

    Nearby supports are first connected by a small morphological dilation and
    softened at their boundaries.  A pixel covered by one scale keeps that
    scale's response; a pixel covered by multiple nearby supports receives a
    support-weighted average.  A final small blur removes seams introduced by
    the region construction without merging distant evidence regions.
    """
    maps = np.stack(scale_maps, axis=0).astype(np.float32)
    supports = np.stack(
        [
            _soft_support_mask(
                heat,
                threshold=threshold,
                transition=transition,
                overlap_radius=overlap_radius,
                blur_radius=blur_radius,
            )
            for heat in maps
        ],
        axis=0,
    )
    weighted_sum = (maps * supports).sum(axis=0)
    support_sum = supports.sum(axis=0)
    fused = np.divide(
        weighted_sum,
        support_sum,
        out=np.zeros_like(support_sum, dtype=np.float32),
        where=support_sum > 1e-6,
    )

    if float(blur_radius) > 0:
        fused_image = Image.fromarray(
            (np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="L",
        ).filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        fused = np.asarray(fused_image, dtype=np.float32) / 255.0

    maximum = float(fused.max())
    if maximum > 1e-6:
        fused = fused / maximum
    return np.clip(fused, 0.0, 1.0)


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    return value.strip("_") or "image"


def class_names(args) -> List[str]:
    if args.class_names:
        names = [item.strip() for item in args.class_names.split(",") if item.strip()]
        if len(names) != args.num_classes:
            raise ValueError("--class_names must contain exactly --num_classes names.")
        return names
    return [f"class_{index}" for index in range(args.num_classes)]


def resolve_original_image(
    name: str,
    dataset_root: Optional[str],
    image_root: Optional[str],
    fallback: torch.Tensor,
) -> Image.Image:
    candidates: List[Path] = []
    name_path = Path(name).expanduser()
    if name_path.is_absolute():
        candidates.append(name_path)
    if dataset_root:
        candidates.append(resolve_path(dataset_root) / name_path)
    if image_root:
        candidates.append(resolve_path(image_root) / name_path)
    candidates.append(name_path)
    for candidate in candidates:
        if candidate.is_file():
            with Image.open(candidate) as image:
                return image.convert("RGB").copy()

    image = fallback.detach().cpu().float()
    mean = torch.tensor(DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(DEFAULT_STD).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255).astype(np.uint8))


def draw_heatmap(
    image: Image.Image,
    heat: np.ndarray,
    output_path: Path,
    args,
) -> None:
    image = image.convert("RGB")
    base = np.asarray(image, dtype=np.float32) / 255.0
    height, width = base.shape[:2]
    heat = resize_heatmap(heat, width, height)
    cmap = plt.get_cmap(args.colormap)
    heat_strength = max(float(args.heat_strength), 0.0)
    heat_for_color = np.clip(heat * heat_strength, 0.0, 1.0)
    rgba = cmap(heat_for_color)
    opacity = np.clip(float(args.alpha), 0.0, 1.0)
    alpha = opacity * heat_for_color
    overlay = base * (1.0 - alpha[..., None]) + rgba[..., :3] * alpha[..., None]
    Image.fromarray(
        np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="RGB",
    ).save(output_path)


def allocate_image_directory(
    output_dir: Path,
    relation_name: str,
    status: str,
    prediction: int,
    image_name: str,
) -> Path:
    """Create a folder that records correctness and predicted label."""
    status_name = f"{status}_pred_{int(prediction)}"
    parent = output_dir / safe_filename(relation_name) / status_name
    parent.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(Path(image_name).stem)
    candidate = parent / stem
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{stem}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def save_relation_heatmaps(
    args,
    output_dir: Path,
    labels: Sequence[str],
    image_name: str,
    fallback_image: torch.Tensor,
    target: int,
    prediction: int,
    confidence: float,
    token_scores: np.ndarray,
    relation_index: int,
    dataset_root: Optional[str],
    status: Optional[str] = None,
) -> Path:
    """Save four maps under relation/status_pred_label/image_name."""
    true_name = labels[target] if 0 <= target < len(labels) else "unknown"
    predicted_name = labels[prediction]
    if status is None:
        if target < 0:
            status = "unknown"
        else:
            status = "correct" if target == prediction else "incorrect"

    image_dir = allocate_image_directory(
        output_dir,
        true_name,
        status,
        prediction,
        image_name,
    )
    base_image = resolve_original_image(
        image_name,
        dataset_root,
        args.image_root,
        fallback_image,
    )

    scale_lengths = [(BASE_GRID - scale + 1) ** 2 for scale in SCALES]
    split_attention: List[np.ndarray] = []
    start = 0
    for length in scale_lengths:
        split_attention.append(token_scores[start : start + length])
        start += length

    raw_maps: List[np.ndarray] = []
    for scale, scores in zip(SCALES, split_attention):
        raw_maps.append(
            project_topk_heatmap(
                scores,
                scale,
                args.top_k,
                args.sigma_factor,
            )
        )
    maps = [normalize_heatmap(raw_map, args.gamma) for raw_map in raw_maps]
    fused = fuse_overlap_aware(
        maps,
        threshold=args.fusion_threshold,
        transition=args.fusion_transition,
        overlap_radius=args.fusion_overlap_radius,
        blur_radius=args.fusion_blur,
    )
    map_specs = (
        ("scale1x1", maps[0]),
        ("scale2x2", maps[1]),
        ("scale4x4", maps[2]),
        ("fused_overlap_average", fused),
    )
    for suffix, heat in map_specs:
        draw_heatmap(
            base_image,
            heat,
            image_dir / f"{suffix}.png",
            args,
        )

    print(
        f"[{relation_index:05d}] {image_name} | true={true_name} ({target}) | "
        f"pred={predicted_name} ({prediction}, {confidence:.4f}) | "
        f"{status} -> {image_dir}"
    )
    return image_dir


def selection_complete(args, selected_correct, selected_incorrect) -> bool:
    if args.mode == "PIPA":
        return (
            selected_correct >= args.pipa_correct
            and selected_incorrect >= args.pipa_incorrect
        )
    return all(
        selected_correct.get(class_index, 0) >= 1
        and selected_incorrect.get(class_index, 0) >= 1
        for class_index in range(args.num_classes)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw four evidence heatmaps for one image or for representative "
            "correct/incorrect samples from the test set."
        )
    )
    parser.add_argument(
        "--network_file",
        default="/home/host/disk/TangWang/train/SRR_Emotion/network_pced.py",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vit_checkpoint", default=None)
    parser.add_argument("--mode", choices=sorted(MODE_NUM_CLASSES), default="PISC_Fine")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument(
        "--input_mode",
        choices=("auto", "single", "dataset"),
        default="auto",
        help="single reads --image_path; dataset selects representative test samples.",
    )
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--single_label", type=int, default=-1)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--annotation_file", default=None)
    parser.add_argument(
        "--dataset_module",
        default="/home/host/disk/TangWang/train/SRR_Emotion/dataset_llm.py",
    )
    parser.add_argument("--image_root", default=None)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipa_correct",
        type=int,
        default=3,
        help="Number of correct PIPA samples to draw in dataset mode.",
    )
    parser.add_argument(
        "--pipa_incorrect",
        type=int,
        default=3,
        help="Number of incorrect PIPA samples to draw in dataset mode.",
    )
    parser.add_argument(
        "--device",
        default="cuda:1" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--heat_strength", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument(
        "--sigma_factor",
        type=float,
        default=0.5,
        help="Gaussian spread as a multiple of the receptive-field width.",
    )
    parser.add_argument(
        "--fusion_threshold",
        type=float,
        default=0.05,
        help="Minimum normalized response treated as active in overlap fusion.",
    )
    parser.add_argument(
        "--fusion_transition",
        type=float,
        default=0.04,
        help="Soft transition width around the fusion threshold.",
    )
    parser.add_argument(
        "--fusion_overlap_radius",
        type=int,
        default=6,
        help="Radius on the 224x224 heatmap for connecting nearby regions.",
    )
    parser.add_argument(
        "--fusion_blur",
        type=float,
        default=2.0,
        help="Gaussian smoothing radius used to remove fusion seams.",
    )
    parser.add_argument("--colormap", default="turbo")
    parser.add_argument("--class_names", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.num_classes = args.num_classes or MODE_NUM_CLASSES[args.mode]
    if args.input_mode == "auto":
        args.input_mode = "single" if args.image_path else "dataset"
    if args.input_mode == "single" and not args.image_path:
        raise ValueError("Single-image mode requires --image_path.")
    if args.input_mode == "dataset" and args.image_path:
        raise ValueError("Do not combine --input_mode dataset with --image_path.")
    if args.num_classes < 1:
        raise ValueError("--num_classes must be positive.")
    if args.top_k < 1:
        raise ValueError("--top_k must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.pipa_correct < 0 or args.pipa_incorrect < 0:
        raise ValueError("PIPA sample counts must be non-negative.")
    if args.sigma_factor <= 0:
        raise ValueError("--sigma_factor must be positive.")
    if args.gamma <= 0:
        raise ValueError("--gamma must be positive.")
    if not 0.0 <= args.fusion_threshold <= 1.0:
        raise ValueError("--fusion_threshold must be between 0 and 1.")
    if args.fusion_transition <= 0:
        raise ValueError("--fusion_transition must be positive.")
    if args.fusion_overlap_radius < 0:
        raise ValueError("--fusion_overlap_radius must be non-negative.")
    if args.fusion_blur < 0:
        raise ValueError("--fusion_blur must be non-negative.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = build_dataset(args)
    model = load_model(args, device)
    labels = class_names(args)

    dataset_root_for_lookup = args.dataset_root or DEFAULT_DATA_ROOT[args.mode]
    loader = DataLoader(
        dataset,
        batch_size=1 if args.input_mode == "single" else args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_relations,
        pin_memory=device.type == "cuda",
    )

    if args.input_mode == "single":
        batch = next(iter(loader))
        logits, token_attention = forward_with_raw_attention(model, batch, device)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        target = int(batch[8][0].item())
        prediction = int(probabilities[0].argmax())
        confidence = float(probabilities[0, prediction])
        image_name = as_name(batch[0][0])
        save_relation_heatmaps(
            args=args,
            output_dir=output_dir,
            labels=labels,
            image_name=image_name,
            fallback_image=batch[1][0],
            target=target,
            prediction=prediction,
            confidence=confidence,
            token_scores=token_attention.numpy()[0],
            relation_index=0,
            dataset_root=dataset_root_for_lookup,
        )
        print(f"Saved one-image heatmaps to {output_dir}")
        return

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    sampled_dataset = Subset(dataset, indices)
    sampled_loader = DataLoader(
        sampled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_relations,
        pin_memory=device.type == "cuda",
    )

    if args.mode == "PIPA":
        selected_correct = 0
        selected_incorrect = 0
    else:
        selected_correct = {class_index: 0 for class_index in range(args.num_classes)}
        selected_incorrect = {class_index: 0 for class_index in range(args.num_classes)}

    saved_count = 0
    stop = False
    for batch in sampled_loader:
        logits, token_attention = forward_with_raw_attention(model, batch, device)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        targets = batch[8].numpy().astype(int)
        counts = batch[-1].numpy().astype(int)
        relation_image_indices = np.repeat(np.arange(len(batch[0])), counts)

        for relation_index, image_index in enumerate(relation_image_indices):
            target = int(targets[relation_index])
            if not 0 <= target < args.num_classes:
                continue
            prediction = int(probabilities[relation_index].argmax())
            is_correct = prediction == target
            status = "correct" if is_correct else "incorrect"

            if args.mode == "PIPA":
                if is_correct:
                    if selected_correct >= args.pipa_correct:
                        continue
                    selected_correct += 1
                else:
                    if selected_incorrect >= args.pipa_incorrect:
                        continue
                    selected_incorrect += 1
            else:
                selected = selected_correct if is_correct else selected_incorrect
                if selected[target] >= 1:
                    continue
                selected[target] += 1

            image_name = as_name(batch[0][int(image_index)])
            save_relation_heatmaps(
                args=args,
                output_dir=output_dir,
                labels=labels,
                image_name=image_name,
                fallback_image=batch[1][int(image_index)],
                target=target,
                prediction=prediction,
                confidence=float(probabilities[relation_index, prediction]),
                token_scores=token_attention.numpy()[relation_index],
                relation_index=saved_count,
                dataset_root=dataset_root_for_lookup,
                status=status,
            )
            saved_count += 1

            if selection_complete(args, selected_correct, selected_incorrect):
                stop = True
                break
        if stop:
            break

    if args.mode == "PIPA":
        print(
            f"Selected PIPA samples: correct={selected_correct}/{args.pipa_correct}, "
            f"incorrect={selected_incorrect}/{args.pipa_incorrect}."
        )
    else:
        missing_correct = [
            labels[index]
            for index, count in selected_correct.items()
            if count == 0
        ]
        missing_incorrect = [
            labels[index]
            for index, count in selected_incorrect.items()
            if count == 0
        ]
        if missing_correct or missing_incorrect:
            print(
                "[warning] No representative sample was found for: "
                f"correct={missing_correct}, incorrect={missing_incorrect}"
            )
    print(f"Saved {saved_count} representative relations to {output_dir}")


if __name__ == "__main__":
    main()
