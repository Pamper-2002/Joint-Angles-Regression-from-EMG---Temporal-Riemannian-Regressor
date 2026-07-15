"""UmeTrack 手部 FK + LBS 蒙皮的对外封装(单帧、实时友好)。

设计目标:给上层渲染器一个极简接口 —— 输入 20 个关节角(弧度),输出
可直接喂给 Open3D 的 (21,3) 关键点 或 (788 顶点, 1544 三角面) 蒙皮网格。

维度约定(源自 emg2pose kinematics.py / visualization.py 的底层调用):
- UmeTrack 的 `_hand_skinning_transform` 只使用 joint_angles[:, 0:20];
  第 20/21 号自由度(手腕)不由关节角控制,而由 wrist_transforms 提供。
- 因此对外只暴露 20 维关节角;内部 pad 到 22 维以对齐手模型张量形状。
- 全局手腕朝向通过 wrist_transforms(4x4 仿射)传入,与 20 个局部关节角解耦。

性能:788 顶点 x 17 骨骼的 LBS 为极小矩阵乘,CPU 单帧 ~1.2ms(实测),
无需 GPU。所有手模型张量在加载时一次性缓存,每帧只做前向计算(torch.no_grad)。
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch

from .hand import HandModel
from .hand_skinning import _skin_points, skin_landmarks

_HAND_MODEL_JSON = os.path.join(os.path.dirname(__file__), "generic_hand_model.json")

# 关节角数量(emg2pose 定义:每指 3 屈曲 FE + 1 外展 AA)
NUM_JOINTS = 20

# 手模型加载后缓存的全局单例(避免每帧重复读盘/建张量)
_HM: Optional[HandModel] = None
_MESH_TRIANGLES_NP: Optional[np.ndarray] = None
_DENSE_W_BATCHED: Optional[torch.Tensor] = None  # dense_bone_weights 预扩 batch 维 (1,V,17)
JOINT_LIMITS: Optional[np.ndarray] = None         # (22,2) 关节角上下限(弧度)


def load_hand_model(json_path: str = _HAND_MODEL_JSON) -> HandModel:
    """加载默认通用手模型并缓存(float32)。参考 emg2pose visualization.load_default_hand_model。"""
    global _HM, _MESH_TRIANGLES_NP, _DENSE_W_BATCHED, JOINT_LIMITS
    if _HM is not None:
        return _HM

    with open(json_path, "r") as fp:
        d = json.load(fp)

    tensors = {}
    for k, v in d.items():
        t = torch.tensor(v)
        # 索引类字段保持整型,其余用 float32(实时计算足够且更快)
        if k in ("joint_frame_index", "joint_parent", "joint_first_child",
                 "joint_next_sibling", "landmark_rest_bone_indices", "mesh_triangles"):
            t = t.long()
        else:
            t = t.float()
        tensors[k] = t

    _HM = HandModel(**tensors)
    _MESH_TRIANGLES_NP = _HM.mesh_triangles.cpu().numpy().astype(np.int32)
    # dense_bone_weights (V,17) -> (1,V,17):_skin_points 的 skin_mat 需带 batch 维
    _DENSE_W_BATCHED = _HM.dense_bone_weights.unsqueeze(0).contiguous()
    JOINT_LIMITS = _HM.joint_limits.cpu().numpy() if _HM.joint_limits is not None else None
    return _HM


def _to_22_tensor(joint_angles_20) -> torch.Tensor:
    """把 20 维关节角(numpy/list/tensor)转成 (22,) float32 张量,末尾补 2 个手腕零角。"""
    ja = torch.as_tensor(np.asarray(joint_angles_20, dtype=np.float32)).reshape(-1)
    if ja.numel() < NUM_JOINTS:
        raise ValueError(f"关节角维度应 >= {NUM_JOINTS}, 实际 {ja.numel()}")
    ja20 = ja[:NUM_JOINTS]
    pad = torch.zeros(2, dtype=torch.float32)
    return torch.cat([ja20, pad], dim=0)  # (22,)


def _identity_wrist() -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32)


@torch.no_grad()
def landmarks_from_angles(joint_angles_20, wrist_transform: Optional[np.ndarray] = None) -> np.ndarray:
    """20 关节角(弧度) -> (21,3) 关键点。用于标定与几何校验。

    wrist_transform: 可选 (4,4) 仿射,提供全局手腕朝向;None 时用单位阵。
    """
    hm = load_hand_model()
    ja = _to_22_tensor(joint_angles_20)  # (22,)
    wrist = _identity_wrist() if wrist_transform is None else \
        torch.as_tensor(np.asarray(wrist_transform, dtype=np.float32)).reshape(4, 4)
    lm = skin_landmarks(hm, ja, wrist_transforms=wrist)  # (21,3)
    return lm.cpu().numpy()


@torch.no_grad()
def mesh_from_angles(joint_angles_20, wrist_transform: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """20 关节角(弧度) -> (顶点 (788,3), 三角面 (1544,3))。渲染主力。

    复刻 emg2pose visualization.skin_vertices 的底层调用(_skin_points + dense_bone_weights),
    但绕开其 av/plotly 依赖,并按单帧场景固定 batch 维=1。
    """
    hm = load_hand_model()
    ja = _to_22_tensor(joint_angles_20)  # (22,)
    wrist = _identity_wrist() if wrist_transform is None else \
        torch.as_tensor(np.asarray(wrist_transform, dtype=np.float32)).reshape(4, 4)

    verts = _skin_points(
        hm.joint_rest_positions,   # (22,3)
        hm.joint_rotation_axes,    # (22,3)
        _DENSE_W_BATCHED,          # (1,788,17)
        ja,                        # (22,)
        hm.mesh_vertices,          # (788,3)
        wrist,                     # (4,4)
    )  # -> (788,3)
    return verts.cpu().numpy(), _MESH_TRIANGLES_NP


def mesh_triangles() -> np.ndarray:
    """返回三角面索引 (1544,3)(拓扑固定不变,渲染器初始化时取一次即可)。"""
    load_hand_model()
    return _MESH_TRIANGLES_NP


def joint_limits() -> np.ndarray:
    """返回 (22,2) 关节角上下限(弧度)。用函数而非模块级变量,避免 import 时机早于加载。"""
    load_hand_model()
    return JOINT_LIMITS
