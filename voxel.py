import numpy as np
import pyrender
import torch
import trimesh
from pyrender.trackball import Trackball

from torch import nn, einsum
import torch.nn.functional as F

# from rlbench.backend.const import DEPTH_SCALE
from scipy.spatial.transform import Rotation
from functools import reduce as funtool_reduce
from operator import mul

# SCALE_FACTOR = DEPTH_SCALE
DEFAULT_SCENE_SCALE = 2.0


def loss_weights(replay_sample, beta=1.0):
    loss_weights = 1.0
    if "sampling_probabilities" in replay_sample:
        probs = replay_sample["sampling_probabilities"]
        loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
        loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
    return loss_weights


def soft_updates(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler("xyz", degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def point_to_voxel_index(
    point: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray
):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32),
        dims_m_one,
    )
    return voxel_indicy


def point_to_pixel_index(
    point: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(
        -intrinsics[0, 0] * (px / pz) + intrinsics[0, 2]
    )
    py = 2 * intrinsics[1, 2] - int(
        -intrinsics[1, 1] * (py / pz) + intrinsics[1, 2]
    )
    return px, py


def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp


def _from_trimesh_scene(trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {
        name: pyrender.Mesh.from_trimesh(geom, smooth=False)
        for name, geom in trimesh_scene.geometry.items()
    }
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def _create_bounding_box(scene, voxel_size, res):
    l = voxel_size * res
    T = np.eye(4)
    w = 0.01
    for trans in [[0, 0, l / 2], [0, l, l / 2], [l, l, l / 2], [l, 0, l / 2]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(
            trimesh.creation.box([w, w, l], T, face_colors=[0, 0, 0, 255])
        )
    for trans in [[l / 2, 0, 0], [l / 2, 0, l], [l / 2, l, 0], [l / 2, l, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(
            trimesh.creation.box([l, w, w], T, face_colors=[0, 0, 0, 255])
        )
    for trans in [[0, l / 2, 0], [0, l / 2, l], [l, l / 2, 0], [l, l / 2, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(
            trimesh.creation.box([w, l, w], T, face_colors=[0, 0, 0, 255])
        )


def create_voxel_scene(
    voxel_grid: np.ndarray,
    q_attention: np.ndarray = None,
    highlight_coordinate: np.ndarray = None,
    highlight_alpha: float = 1.0,
    voxel_size: float = 0.1,
    show_bb: bool = False,
    alpha: float = 0.5,
):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(
        np.full_like(occupancy, alpha, dtype=np.float32), -1
    )
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1) / 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = q > 0.75
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate(
            [q, np.zeros_like(q), np.zeros_like(q), np.clip(q, 0, 1)], axis=-1
        )
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0)
    )
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform
    )
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    if show_bb:
        assert d == h == w
        _create_bounding_box(scene, voxel_size, d)
    return scene


def visualise_voxel(
    voxel_grid: np.ndarray,
    q_attention: np.ndarray = None,
    highlight_coordinate: np.ndarray = None,
    highlight_alpha: float = 1.0,
    rotation_amount: float = 0.0,
    show: bool = False,
    voxel_size: float = 0.1,
    offscreen_renderer: pyrender.OffscreenRenderer = None,
    show_bb: bool = False,
):
    scene = create_voxel_scene(
        voxel_grid,
        q_attention,
        highlight_coordinate,
        highlight_alpha,
        voxel_size,
        show_bb,
    )
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=640, viewport_height=480, point_size=1.0
        )
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8], bg_color=[1.0, 1.0, 1.0]
        )
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height
        )
        p = _compute_initial_camera_pose(s)
        t = Trackball(
            p, (r.viewport_width, r.viewport_height), s.scale, s.centroid
        )
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)
        color, depth = r.render(s)
        return color.copy()


MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False


class VoxelGrid(nn.Module):
    def __init__(
        self,
        coord_bounds,
        voxel_size: int,
        device,
        batch_size,
        feature_size,
        max_num_coords: int,
    ):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = (
            torch.tensor(self._voxel_shape, device=device).unsqueeze(0) + 2
        )  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(
            coord_bounds, dtype=torch.float, device=device
        ).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [
                torch.tensor([batch_size], device=device),
                max_dims,
                torch.tensor([4 + feature_size], device=device),
            ],
            -1,
        ).tolist()
        self._ones_max_coords = torch.ones(
            (batch_size, max_num_coords, 1), device=device
        )
        self._num_coords = max_num_coords

        shape = self._total_dims_list

        self._result_dim_sizes = torch.tensor(
            [
                funtool_reduce(mul, shape[i + 1 :], 1)
                for i in range(len(shape) - 1)
            ]
            + [1],
            device=device,
        )
        flat_result_size = funtool_reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float, device=device)
        self._flat_output = (
            torch.ones(flat_result_size, dtype=torch.float, device=device)
            * self._initial_val
        )
        self._arange_to_max_coords = torch.arange(
            4 + feature_size, device=device
        )
        self._flat_zeros = torch.zeros(
            flat_result_size, dtype=torch.float, device=device
        )

        self._const_1 = torch.tensor(1.0, device=device)
        self._batch_size = batch_size

        # Coordinate Bounds:
        self._bb_mins = self._coord_bounds[..., 0:3]
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - self._bb_mins
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int()
        self._dims_orig = dims_orig = self._voxel_shape_spec.int() - 2
        self._dims_m_one = (dims - 1).int()
        # BS x 1 x 3
        self._res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)

        self._voxel_indicy_denmominator = self._res + MIN_DENOMINATOR
        self._dims_m_one_zeros = torch.zeros_like(self._dims_m_one)

        batch_indices = torch.arange(
            self._batch_size, dtype=torch.int, device=device
        ).view(self._batch_size, 1, 1)
        self._tiled_batch_indices = batch_indices.repeat(
            [1, self._num_coords, 1]
        )

        w = self._voxel_shape[0] + 2
        arange = torch.arange(0, w, dtype=torch.float, device=device)
        self._index_grid = (
            torch.cat(
                [
                    arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
                    arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
                    arange.view(1, 1, w, 1).repeat([w, w, 1, 1]),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .repeat([self._batch_size, 1, 1, 1, 1])
        )

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        out: torch.Tensor,
        dim: int = -1,
    ):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims]
        )
        indices_for_flat_tiled = (
            ((indices * indices_scales).sum(dim=-1, keepdims=True))
            .view(-1, 1)
            .repeat(*[1, self._voxel_feature_size])
        )

        implicit_indices = (
            self._arange_to_max_coords[: self._voxel_feature_size]
            .unsqueeze(0)
            .repeat(*[indices_for_flat_tiled.shape[0], 1])
        )
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates,
            flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output),
        )
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(
        self, coords, coord_features=None, coord_bounds=None
    ):
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR

        bb_mins_shifted = bb_mins - res  # shift back by one
        floor = torch.floor(
            (coords - bb_mins_shifted.unsqueeze(1))
            / voxel_indicy_denmominator.unsqueeze(1)
        ).int()
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)

        # global-coordinate point cloud (x, y, z)
        voxel_values = coords

        # rgb values (R, G, B)
        if coord_features is not None:
            voxel_values = torch.cat(
                [voxel_values, coord_features], -1
            )  # concat rgb values (B, 128, 128, 3)

        # coordinates to aggregate over
        _, num_coords, _ = voxel_indices.shape
        all_indices = torch.cat(
            [self._tiled_batch_indices[:, :num_coords], voxel_indices], -1
        )

        # max coordinates
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1
        )

        # aggregate across camera views
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size),
        )

        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (
                res_centre
                + bb_mins_shifted.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            )[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat(
                [vox[..., :-1], coord_positions, vox[..., -1:]], -1
            )

        # occupied value
        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([vox[..., :-1], occupied], -1)

        # hard voxel-location position encoding
        return torch.cat(
            [
                vox[..., :-1],
                self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
                vox[..., -1:],
            ],
            -1,
        )
