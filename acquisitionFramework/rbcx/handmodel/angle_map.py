"""角度语义映射层:MediaPipe 几何角(度) -> UmeTrack FK 关节角(弧度)。

这是整个方案的技术核心。两种角度语义完全不同,必须转换。

【关键教训 / 为何用双锚点而非统一 180-mp】
早期版本对所有 FE 关节用 flex=180-mp 的统一假设,导致"张开手却显示半握"。
根因(已实测):MediaPipe 的关节角计算里,只有 PIP/DIP 在手指伸直时≈180°;
而 MCP_FE 用的是 _angle_3pts(掌心, 指根, 中节),伸直时基准角只有 ~110-160°
(且各指不同)。若仍套用 180-mp,MCP 会被误当成弯曲 ~50°,手指就并拢半握。

【双锚点线性映射模型】
对每个关节定义两个 MediaPipe 锚点:
    open_deg  : 手完全张开(伸直)时该关节的 mediapipe 角
    close_deg : 手完全握拳时该关节的 mediapipe 角
把当前 mediapipe 角在 [open_deg, close_deg] 上归一化成 t∈[0,1](0=张开,1=握拳),
再线性映射到 UmeTrack 该关节的 [rest, max_flex]:
    FE : u_rad = t * limit_hi              (张开 t=0 -> 0 rad 伸直; 握拳 t=1 -> 上限)
    AA : u_rad = (mp_deg - open_deg) * k_aa (相对张开基准的带符号偏差)
最后按 joint_limits 截断。这保证"张开手 -> 伸直"精确成立,根治半握问题。

锚点有解析默认(来自实测平手 + 解剖常识),可被标定脚本(calibrate_angle_map.py)
拟合出的 scale/offset 覆盖精修(两者结合策略)。

用法:
    m = AngleMapper()                 # 自动尝试加载标定参数,失败则用解析锚点
    u20 = m.map(mediapipe_angles_20)  # -> (20,) float32 弧度,已截断到 joint_limits
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from .fk import joint_limits, NUM_JOINTS

# 与 emg2pose / 本项目 mediapipe.py 完全一致的 20 关节顺序
JOINT_NAMES = [
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
]

# 每个关节是否为屈曲角(FE)。非 FE 即外展角(AA)。
IS_FE = np.array([("_FE" in n) for n in JOINT_NAMES], dtype=bool)

_PARAMS_JSON = os.path.join(os.path.dirname(__file__), "angle_map_params.json")
_DEG2RAD = np.pi / 180.0

# ---- FE 关节的双锚点(度):手张开伸直 open_deg,握拳 close_deg ----
# 来源:实测张开平手(PIP/DIP≈180、MCP 掌心夹角≈110~160)+ 握拳解剖近似。
# MCP 用掌心参考,对屈曲不敏感、动态范围窄,故 open/close 间隔较小;PIP/DIP 范围大。
# MCP_FE 用掌心参考,张开基准角各指差异大(实测 index~110 mid~158 ring~117 pinky~103)
# 且对屈曲不敏感(握拳才变 ~17°)。故 MCP 的 open 取各指实测张开值,close 只比 open 略小,
# 使 MCP 只贡献小幅屈曲(手指弯曲的视觉主要由 PIP/DIP 承担),避免张开时被误判半握。
_FE_OPEN = {
    "THUMB_CMC_FE": 150.0, "THUMB_MCP_FE": 165.0, "THUMB_IP_FE": 178.0,
    "INDEX_MCP_FE": 112.0, "INDEX_PIP_FE": 178.0, "INDEX_DIP_FE": 178.0,
    "MIDDLE_MCP_FE": 158.0, "MIDDLE_PIP_FE": 178.0, "MIDDLE_DIP_FE": 178.0,
    "RING_MCP_FE": 118.0, "RING_PIP_FE": 178.0, "RING_DIP_FE": 178.0,
    "PINKY_MCP_FE": 104.0, "PINKY_PIP_FE": 178.0, "PINKY_DIP_FE": 178.0,
}
_FE_CLOSE = {
    "THUMB_CMC_FE": 115.0, "THUMB_MCP_FE": 110.0, "THUMB_IP_FE": 120.0,
    "INDEX_MCP_FE": 90.0, "INDEX_PIP_FE": 95.0, "INDEX_DIP_FE": 110.0,
    "MIDDLE_MCP_FE": 130.0, "MIDDLE_PIP_FE": 95.0, "MIDDLE_DIP_FE": 110.0,
    "RING_MCP_FE": 95.0, "RING_PIP_FE": 95.0, "RING_DIP_FE": 110.0,
    "PINKY_MCP_FE": 80.0, "PINKY_PIP_FE": 95.0, "PINKY_DIP_FE": 110.0,
}
# ---- AA 关节:张开时的基准角(度)。相对它取带符号偏差再缩放。 ----
_AA_OPEN = {
    "THUMB_CMC_AA": -25.0, "INDEX_MCP_AA": 18.0, "MIDDLE_MCP_AA": 6.0,
    "RING_MCP_AA": -6.0, "PINKY_MCP_AA": -18.0,
}
_AA_SCALE = 1.0  # mediapipe 外展度 -> UmeTrack 弧度的经验比例(再按 limit 截断)。
                 # 取 1.0 让张开手指更充分分开;真实检测的 AA 幅度通常比几何合成大。


def _build_anchor_arrays():
    """构造逐关节锚点数组。返回 (open_deg(20), close_deg(20), aa_open(20))。"""
    open_deg = np.zeros(NUM_JOINTS)
    close_deg = np.zeros(NUM_JOINTS)
    aa_open = np.zeros(NUM_JOINTS)
    for i, n in enumerate(JOINT_NAMES):
        if IS_FE[i]:
            open_deg[i] = _FE_OPEN[n]
            close_deg[i] = _FE_CLOSE[n]
        else:
            aa_open[i] = _AA_OPEN.get(n, 0.0)
    return open_deg, close_deg, aa_open


class AngleMapper:
    def __init__(self, params_path: str = _PARAMS_JSON):
        self.limits = joint_limits()[:NUM_JOINTS].astype(np.float64)  # (20,2)
        self.open_deg, self.close_deg, self.aa_open = _build_anchor_arrays()
        self.aa_sign = np.ones(NUM_JOINTS, dtype=np.float64)  # AA 方向(标定可翻转)
        self.aa_scale = np.full(NUM_JOINTS, _AA_SCALE, dtype=np.float64)
        # 可选的标定精修增益(默认恒等):对最终弧度做 gain*u + bias 微调
        self.scale = np.ones(NUM_JOINTS, dtype=np.float64)
        self.offset = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.calibrated = False
        self._try_load(params_path)

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as fp:
                d = json.load(fp)
            for k in ("open_deg", "close_deg", "aa_open", "aa_sign",
                      "aa_scale", "scale", "offset"):
                if k in d:
                    setattr(self, k, np.asarray(d[k], dtype=np.float64))
            self.calibrated = True
        except Exception as e:
            print(f"[AngleMapper] 标定参数加载失败,回退解析锚点: {e}")

    def map(self, mediapipe_angles_20) -> np.ndarray:
        """MediaPipe 20 角(度) -> UmeTrack 20 关节角(弧度),已截断到 joint_limits。"""
        mp = np.asarray(mediapipe_angles_20, dtype=np.float64).reshape(-1)[:NUM_JOINTS]
        u = np.zeros(NUM_JOINTS, dtype=np.float64)

        # FE: 在 [open, close] 上归一化 t∈[0,1](0 张开, 1 握拳) -> [0, limit_hi]
        fe = IS_FE
        span = (self.close_deg - self.open_deg)
        span[np.abs(span) < 1e-6] = 1e-6
        t = (mp - self.open_deg) / span            # 0=张开 1=握拳
        t = np.clip(t, 0.0, 1.5)                    # 允许略超上限,由 clip 兜底
        u[fe] = t[fe] * self.limits[fe, 1]          # limit_hi = 最大屈曲

        # AA: 相对张开基准的带符号偏差 * 缩放 * 方向
        aa = ~fe
        u[aa] = (mp[aa] - self.aa_open[aa]) * _DEG2RAD * self.aa_scale[aa] * self.aa_sign[aa]

        # 标定精修增益(默认恒等)
        u = self.scale * u + self.offset
        # 截断到物理关节限位
        u = np.clip(u, self.limits[:, 0], self.limits[:, 1])
        return u.astype(np.float32)

    def save(self, path: str = _PARAMS_JSON) -> None:
        with open(path, "w") as fp:
            json.dump({
                "open_deg": self.open_deg.tolist(),
                "close_deg": self.close_deg.tolist(),
                "aa_open": self.aa_open.tolist(),
                "aa_sign": self.aa_sign.tolist(),
                "aa_scale": self.aa_scale.tolist(),
                "scale": self.scale.tolist(),
                "offset": self.offset.tolist(),
                "joint_names": JOINT_NAMES,
            }, fp, indent=2)
