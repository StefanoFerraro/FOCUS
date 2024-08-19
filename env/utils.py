from dm_env import specs
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import os
import requests
import gdown
import wget
import skimage

def segment_image(image, bbox):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.shape[:2], (255, 255, 255))
    transparency_mask = np.zeros(
        (image_array.shape[0], image_array.shape[1]), dtype=np.uint8
    )
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def filter_masks(annotations):  # filte the overlap mask
    annotations.sort(key=lambda x: x["area"], reverse=True)
    to_remove = set()
    for i in range(0, len(annotations)):
        a = annotations[i]
        for j in range(i + 1, len(annotations)):
            b = annotations[j]
            if i != j and j not in to_remove:
                # check if
                if b["area"] < a["area"]:
                    if (a["segmentation"] & b["segmentation"]).sum() / b[
                        "segmentation"
                    ].sum() > 0.8:
                        to_remove.add(j)

    return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove


def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
        h = y2 - y1
        w = x2 - x1
    return [x1, y1, x2, y2]


#   CPU post process
def fast_show_mask(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    points=None,
    pointlabel=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color == True:
        color = np.random.random((msak_sum, 1, 1, 3))
    else:
        color = np.ones((msak_sum, 1, 1, 3)) * np.array(
            [30 / 255, 144 / 255, 255 / 255]
        )
    transparency = np.ones((msak_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    show = np.zeros((height, weight, 4))
    h_indices, w_indices = np.meshgrid(
        np.arange(height), np.arange(weight), indexing="ij"
    )
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1
            )
        )
    # draw point
    if points is not None:
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
            s=20,
            c="y",
        )
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
            s=20,
            c="m",
        )

    if retinamask == False:
        show = cv2.resize(
            show, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
    ax.imshow(show)


def fast_show_mask_gpu(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    points=None,
    pointlabel=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color == True:
        color = torch.rand((msak_sum, 1, 1, 3), device=annotation.device)
    else:
        color = torch.ones((msak_sum, 1, 1, 3), device=annotation.device) * torch.tensor(
            [30 / 255, 144 / 255, 255 / 255]
        , device=annotation.device)
    transparency = torch.ones((msak_sum, 1, 1, 1), device=annotation.device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    show = torch.zeros((height, weight, 4), device=annotation.device)
    h_indices, w_indices = torch.meshgrid(
        torch.arange(height), torch.arange(weight), indexing="ij"
    )
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]
    show_cpu = show.cpu().numpy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1
            )
        )
    # draw point
    if points is not None:
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
            s=20,
            c="y",
        )
        plt.scatter(
            [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
            [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
            s=20,
            c="m",
        )
    if retinamask == False:
        show_cpu = cv2.resize(
            show_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
    ax.imshow(show_cpu)

def crop_image(annotations, image):
    cropped_boxes = []
    cropped_images = []
    not_crop = []
    filter_id = []

    for _, mask in enumerate(annotations):
        if np.sum(mask["segmentation"]) <= 100:
            filter_id.append(_)
            continue
        bbox = get_bbox_from_mask(mask["segmentation"])
        cropped_boxes.append(segment_image(image, bbox))
        cropped_images.append(bbox)

    return cropped_boxes, cropped_images, not_crop, filter_id, annotations


# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def download_checkpoint_wget(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        print("download checkpoints ......")
        wget.download(url, folder)
        print("download successfully!")

    return filepath


def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(
            "Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)"
        )
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath


def obs_specs(obs_space):
    obs_keys = ["rgb", "depth", "proprio", "objects_pos", "segmentation"]
    obs_specs = []
    for k, v in obs_space.items():
        if k in obs_keys:
            obs_specs.append(
                specs.BoundedArray(
                    name=k,
                    shape=v.shape,
                    dtype=v.dtype,
                    minimum=v.low,
                    maximum=v.high,
                )
            )
    return obs_specs


def image_segmentation(image, seg):
    """
    Segmentation of image based on segmentation mask, works both with rgb and depth images.
    seg: (dim1, dim2, channels)
    image: (dim1, dim2, 1/3)

    Return:
    seg_image: (dim1, dim2, 1/3, channels)
    """

    seg_chs, _, _ = seg.shape
    image_chs, _, _ = image.shape

    seg_mask = np.repeat(np.expand_dims(seg, axis=1), image_chs, 1)
    seg_image = np.zeros((seg_chs, *image.shape), dtype=image.dtype)

    for ch in range(seg_chs):
        seg_image[ch] = image * seg_mask[ch]

    return seg_image

def custom_target_2d(obs, target_pos, diam, env):
    if diam == 1:
        target_pixels = [np.array(int(x * 100) + 32) for x in target_pos[::-1]] # single pixel target
    else:
        target_pixels = skimage.draw.disk((target_pos[::-1] * [-100, 100] + [32, 32]), radius=int(diam/2))
    
    torch_to_numpy = lambda x: x.detach().cpu().numpy() if type(x) == torch.Tensor else x

    obs["rgb"][... ,0,*target_pixels] = 255
    obs["rgb"][... ,1,*target_pixels] = 0
    obs["rgb"][... ,2,*target_pixels] = 0
    # adapt proprio and reward to the new target
    coord_dist_to_target = torch_to_numpy(obs["objects_pos"][...,0,:]) - target_pos
    dist_to_target = np.sqrt(np.sum((coord_dist_to_target)**2, axis=-1))
    # rewards = env.get_reward(dist_to_target)
    
    # provide directly the target position in proprio and not the distance between target and eef
    obs["proprio"][...,2:4] = torch.tensor(target_pos, device="cuda") if type(obs["proprio"]) == torch.Tensor else target_pos
    # obs["reward"] = torch.tensor(rewards, device="cuda").unsqueeze(-1) if type(obs["reward"]) == torch.Tensor else rewards
                    
    return obs