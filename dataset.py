import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pickle as pkl
import math
from PIL import ImageDraw, ImageFilter


def compute_spatial_configuration(
    p1_box,
    p2_box,
    image_width,
    image_height,
    eps=1e-6
):


    x1_a, y1_a, x2_a, y2_a = map(float, p1_box)
    x1_b, y1_b, x2_b, y2_b = map(float, p2_box)

    w_a = max(x2_a - x1_a, eps)
    h_a = max(y2_a - y1_a, eps)

    w_b = max(x2_b - x1_b, eps)
    h_b = max(y2_b - y1_b, eps)

    cx_a = (x1_a + x2_a) / 2.0
    cy_a = (y1_a + y2_a) / 2.0

    cx_b = (x1_b + x2_b) / 2.0
    cy_b = (y1_b + y2_b) / 2.0

    delta_x = (cx_a - cx_b) / image_width
    delta_y = (cy_a - cy_b) / image_height


    log_width_ratio = math.log(w_a / w_b)
    log_height_ratio = math.log(h_a / h_b)


    distance = math.sqrt(
        delta_x ** 2 + delta_y ** 2
    )

    # IoU
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = w_a * h_a
    area_b = w_b * h_b
    union = area_a + area_b - intersection

    iou = intersection / max(union, eps)

    return [
        delta_x,
        delta_y,
        log_width_ratio,
        log_height_ratio,
        distance,
        iou
    ]


def _mask_to_tensor(mask, size=(224, 224)):
    """
    Convert a PIL mask to a float tensor with shape (1, H, W).
    Nearest-neighbor interpolation preserves binary regions.
    """
    mask = mask.resize(size, Image.Resampling.NEAREST)
    mask = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(mask).unsqueeze(0)


def _make_box_mask(image_size, boxes):
    """
    Create a binary mask from one or more boxes.

    Args:
        image_size: (width, height)
        boxes: list of [x1, y1, x2, y2]

    Returns:
        PIL.Image in mode "L"
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        draw.rectangle(
            [int(v) for v in box],
            fill=255,
        )

    return mask


def adjust_box_coordinates(box):
    """
    Adjust the box coordinates to ensure left < right and upper < lower.

    Parameters:
    box (tuple or list): A tuple or list of 4 elements representing the coordinates (left, upper, right, lower).

    Returns:
    tuple: A tuple with the adjusted coordinates.
    """
    left, upper, right, lower = box
    # Ensure left < right
    if left > right:
        left, right = right, left
    # Ensure upper < lower
    if upper > lower:
        upper, lower = lower, upper
    return left, upper, right, lower


class DataSet(data.Dataset):
    def __init__(self, image_dir, data_dir, input_transform, full_im_transform):

        self.input_transform = input_transform
        self.full_im_transform = full_im_transform

        self.image_dir = image_dir
        self.person_box = []
        self.face_box = []
        self.person_pair = []
        self.rel_classes = []
        self.img_names = []

        data = pkl.load(open(data_dir, "rb"), encoding='latin1')

        for item in data:
            self.img_names.append(item['img'])

            self.person_box.append(item['Pbboxs'])

            self.person_pair.append(item['person_pair'])

            self.rel_classes.append(item['rel'])

    def __getitem__(self, index):

        img = Image.open(
            os.path.join(
                self.image_dir,
                str(self.img_names[index])
            )
        ).convert("RGB")
        person_box = self.person_box[index]

        ## target
        rel_classes = self.rel_classes[index]
        person_pair = self.person_pair[index]

        rel_num = len(rel_classes)

        person_box1 = torch.empty(rel_num, 3, 224, 224)
        person_box2 = torch.empty(rel_num, 3, 224, 224)
        union_box = torch.empty(rel_num, 3, 224, 224)

        pair_mask = torch.empty(rel_num, 1, 224, 224)
        interaction_mask = torch.empty(rel_num, 1, 224, 224)
        bposes = np.zeros((rel_num, 6), dtype=np.float32)

     
        if self.full_im_transform is not None:
            full_im = self.full_im_transform(img)
        else:
            full_im = self.input_transform(img)

        for rel in range(rel_num):
            p1 = person_pair[rel, 0]
            p2 = person_pair[rel, 1]

            p1_box = adjust_box_coordinates(person_box[p1])
            p2_box = adjust_box_coordinates(person_box[p2])

            p1_crop = img.crop(
                (
                    int(p1_box[0]),
                    int(p1_box[1]),
                    int(p1_box[2]),
                    int(p1_box[3]),
                )
            )

            p2_crop = img.crop(
                (
                    int(p2_box[0]),
                    int(p2_box[1]),
                    int(p2_box[2]),
                    int(p2_box[3]),
                )
            )

            person_box1[rel] = self.input_transform(p1_crop)
            person_box2[rel] = self.input_transform(p2_crop)

            u_x1 = min(p1_box[0], p2_box[0])
            u_y1 = min(p1_box[1], p2_box[1])
            u_x2 = max(p1_box[2], p2_box[2])
            u_y2 = max(p1_box[3], p2_box[3])

            union_crop = img.crop(
                (
                    int(u_x1),
                    int(u_y1),
                    int(u_x2),
                    int(u_y2),
                )
            )
            union_box[rel] = self.input_transform(union_crop)

         
            target_mask = _make_box_mask(
                image_size=img.size,
                boxes=[p1_box, p2_box],
            )

          
            union_region_mask = _make_box_mask(
                image_size=img.size,
                boxes=[[u_x1, u_y1, u_x2, u_y2]],
            )

            target_mask_np = np.asarray(
                target_mask,
                dtype=np.float32,
            )

            union_region_mask_np = np.asarray(
                union_region_mask,
                dtype=np.float32,
            )

        
            interaction_mask_np = np.maximum(
                union_region_mask_np - target_mask_np,
                0.0,
            )

            interaction_mask_pil = Image.fromarray(
                interaction_mask_np.astype(np.uint8),
                mode="L",
            )

            pair_mask[rel] = _mask_to_tensor(target_mask)
            interaction_mask[rel] = _mask_to_tensor(
                interaction_mask_pil
            )

            spatial_config = compute_spatial_configuration(
                p1_box=p1_box,
                p2_box=p2_box,
                image_width=img.width,
                image_height=img.height
            )

            bposes[rel] = np.asarray(
                spatial_config,
                dtype=np.float32,
            )

        bposes = torch.from_numpy(bposes)

        rel_classes = torch.as_tensor(
            rel_classes,
            dtype=torch.long,
        )

        pair_indices = torch.as_tensor(
            person_pair,
            dtype=torch.long,
        )

        img_rel_num = torch.tensor(
            [rel_num],
            dtype=torch.long,
        )

        return (self.img_names[index], full_im, union_box, person_box1, person_box2, pair_mask, interaction_mask,
                bposes, rel_classes, pair_indices, img_rel_num)

    def __len__(self):
        return len(self.img_names)

