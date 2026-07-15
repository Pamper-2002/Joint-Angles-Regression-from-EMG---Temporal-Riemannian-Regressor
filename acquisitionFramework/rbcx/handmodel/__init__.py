"""UmeTrack 手部前向运动学 + 线性混合蒙皮(LBS)内核(自包含 vendor 版)。

从 emg2pose(NeurIPS 2024)/UmeTrack 精简移植而来,去掉了 lightning/plotly/av
等训练与可视化依赖,只保留“20 关节角 -> 21 关键点 / 788 顶点蒙皮网格”所需的最小内核。

对外主要接口见 fk.py:
    - load_hand_model()          加载默认通用手模型(generic_hand_model.json)
    - landmarks_from_angles()    20 角 -> (21,3) 关键点(用于标定/校验)
    - mesh_from_angles()         20 角 -> (verts(788,3), tris(1544,3)) 蒙皮网格(渲染主力)
"""
from .fk import (
    load_hand_model,
    landmarks_from_angles,
    mesh_from_angles,
    mesh_triangles,
    joint_limits,
)

__all__ = [
    "load_hand_model",
    "landmarks_from_angles",
    "mesh_from_angles",
    "mesh_triangles",
    "joint_limits",
]
