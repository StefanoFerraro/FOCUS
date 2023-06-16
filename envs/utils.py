from dm_env import specs
import numpy as np
import gym


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

def common_obs_space(size, segmentation_instances, include_background=True):
    spaces = {
        "rgb": gym.spaces.Box(0, 255, (3,) + size, dtype=np.uint8),
        "depth": gym.spaces.Box(
            -np.inf,
            np.inf,
            (1,) + size,
            dtype=np.float32,
        ),
        "objects_pos": gym.spaces.Box(
            -2, 2, (len(segmentation_instances), 3), dtype=np.float32
        ),
        "segmentation": gym.spaces.Box(
            0,
            1,
            (len(segmentation_instances) + include_background,) + size,
            dtype=np.uint8,
        ),
        "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
        "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
        "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        "success": gym.spaces.Box(0, 1, (), dtype=bool),
    }
    return spaces

def pixel_to_world(seg, depth):
    depth = depth.transpose(1, 2, 0)

    # depth_map = CU.get_real_depth_map(sim=self._env.sim, depth_map=depth)
    estimated_obj_pos = []

    for ch in range(len(self.segmentation_instances)):
        seg_pixels = np.argwhere(seg[ch])
        if (
            seg_pixels.size > self.objects_pixels[ch]
        ):  # update max number of pixels that compose the objects
            self.objects_pixels[ch] = seg_pixels.size

        if (
            seg_pixels.size > 0.4 * self.objects_pixels[ch]
        ):  # at least 40% of pixels needs to be in view to update the object position
            centroid = np.mean(seg_pixels, axis=0).astype(int)
            estimated_obj_pos += [
                CU.transform_from_pixels_to_world(
                    pixels=centroid,
                    depth_map=depth,
                    camera_to_world_transform=self.camera_to_world,
                )
            ]

        else:  # if object is not detected in the scene just take the last relevand position
            estimated_obj_pos += [self.last_estimated_obj_pos[ch]]

    self.last_estimated_obj_pos = estimated_obj_pos

    return estimated_obj_pos
