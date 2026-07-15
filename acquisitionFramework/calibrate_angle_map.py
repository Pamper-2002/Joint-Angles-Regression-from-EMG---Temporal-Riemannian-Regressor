"""角度映射标定脚本(一次性运行,精修 mediapipe->UmeTrack 的 scale/offset)。

背景:
    angle_map.AngleMapper 的解析初值已能给出合理手形,但 mediapipe 几何角与
    UmeTrack 关节角是不同的参数化,逐关节的比例/零点存在系统偏差。本脚本采集
    一段真实手部运动,以“FK(map(mp_angle)) 的骨段方向 与 mediapipe landmarks
    的骨段方向 尽量一致”为目标,用最小二乘拟合每关节 (scale, offset),写入
    angle_map_params.json 供实时渲染加载。

为何用“骨段方向”而非“关键点坐标”做目标:
    方向对全局平移/缩放/手腕朝向不敏感,只反映关节弯曲本身,标定更鲁棒、无需
    先做刚体对齐。UmeTrack landmarks 与 mediapipe landmarks 的索引经映射后逐指对应。

用法:
    在 acquisitionFramework 目录用 .venv-emg 的 python 运行:
        python calibrate_angle_map.py --collect     # 采集(镜头前做:张开/握拳/逐指弯曲/张指)
        python calibrate_angle_map.py --fit         # 拟合并保存参数
        python calibrate_angle_map.py --collect --fit  # 采完直接拟合
采集时按 q/ESC 结束(RGB 窗口需有焦点)。
"""
import argparse
import os
import time

import numpy as np
from scipy.optimize import least_squares

from rbcx.handtracker.mediapipe import MediaPipeHandTracker
from rbcx.handmodel import landmarks_from_angles, joint_limits
from rbcx.handmodel.angle_map import AngleMapper, JOINT_NAMES, IS_FE, NUM_JOINTS
from rbcx.handmodel.pose_geometry import canonicalize_mediapipe_landmarks

_CALIB_ANGLES = "savedData/_calib_mp_angles.npy"      # (N,20) mediapipe 角(度)
_CALIB_LANDMARKS = "savedData/_calib_mp_landmarks.npy"  # (N,21,3) mediapipe world landmarks

# UmeTrack landmark 索引(见 hand.py LANDMARK 枚举):5 个指尖 + 手腕
# 顺序: 0 thumb_tip,1 index_tip,2 mid_tip,3 ring_tip,4 pinky_tip,5 wrist
UME_TIPS = [0, 1, 2, 3, 4]
UME_WRIST = 5
# MediaPipe 对应指尖: thumb=4,index=8,middle=12,ring=16,pinky=20; wrist=0
MP_TIPS = [4, 8, 12, 16, 20]
MP_WRIST = 0


def collect():
    os.makedirs("savedData", exist_ok=True)
    tracker = MediaPipeHandTracker(camera_index=0, mirror=False, show_window=True,
                                   use_2D_coord_for_angles=True)
    tracker.start()
    angles_buf, lm_buf = [], []
    print("开始采集。请在镜头前依次:完全张开 -> 慢慢握拳 -> 逐指弯曲 -> 张开五指外展。按 q/ESC 结束。")
    try:
        while True:
            st = tracker.get_hand_state("Right")
            lm = st.get("landmarks", None)
            ang = st.get("angles_list", None)
            if lm is not None and ang is not None and np.asarray(lm).shape == (21, 3):
                angles_buf.append(np.asarray(ang, dtype=np.float64))
                lm_buf.append(np.asarray(lm, dtype=np.float64))
            if tracker.show_window and not tracker.poll_gui():
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
    angles = np.array(angles_buf)
    lms = np.array(lm_buf)
    np.save(_CALIB_ANGLES, angles)
    np.save(_CALIB_LANDMARKS, lms)
    print(f"采集完成:{len(angles)} 帧,已存 {_CALIB_ANGLES} / {_CALIB_LANDMARKS}")


def _finger_dir_vectors_mp(lm, reference_landmarks=None):
    """MediaPipe landmarks -> palm-aligned wrist-to-tip directions."""
    if reference_landmarks is not None:
        canonical = canonicalize_mediapipe_landmarks(lm, reference_landmarks, "Right")
        if not canonical.valid:
            raise ValueError(f"invalid calibration landmarks: {canonical.reason}")
        wrist = canonical.landmarks[UME_WRIST]
        points = canonical.landmarks
        tips = UME_TIPS
    else:
        wrist = lm[MP_WRIST]
        points = lm
        tips = MP_TIPS
    dirs = []
    for tip in tips:
        v = points[tip] - wrist
        dirs.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(dirs)


def _finger_dir_vectors_ume(u_angles):
    """UmeTrack FK landmarks -> 5 指的 (wrist->tip) 单位方向 (5,3)。"""
    lm = landmarks_from_angles(u_angles)  # (21,3)
    wrist = lm[UME_WRIST]
    dirs = []
    for tip in UME_TIPS:
        v = lm[tip] - wrist
        dirs.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(dirs)


def fit():
    if not (os.path.exists(_CALIB_ANGLES) and os.path.exists(_CALIB_LANDMARKS)):
        print("缺少采集数据,请先运行 --collect")
        return
    mp_angles = np.load(_CALIB_ANGLES)      # (N,20) 度
    mp_lms = np.load(_CALIB_LANDMARKS)      # (N,21,3)
    N = len(mp_angles)
    if N < 30:
        print(f"警告:仅 {N} 帧,标定可能不稳。建议 >=200 帧。")

    # 下采样以加速(方向目标平滑,无需全量)
    step = max(1, N // 400)
    idx = np.arange(0, N, step)
    mp_angles, mp_lms = mp_angles[idx], mp_lms[idx]
    M = len(mp_angles)

    m = AngleMapper()

    # ---- 数据驱动标定 FE 锚点(open_deg / close_deg) ----
    # 思路:采集应覆盖“完全张开 -> 完全握拳”。对每个 FE 关节,取其 mediapipe 角在整段
    # 数据里的稳健分位数作为两个锚点:伸直端(张开)取高分位,屈曲端(握拳)取低分位。
    # 这直接消除了手写锚点的不准,是本标定最核心、最鲁棒的一步(无需迭代优化)。
    p_lo, p_hi = np.percentile(mp_angles, [5, 95], axis=0)  # (20,),(20,)
    for i in range(NUM_JOINTS):
        if IS_FE[i]:
            # FE 角:张开时角大(伸直≈180),握拳时角小 => open=高分位, close=低分位
            m.open_deg[i] = p_hi[i]
            m.close_deg[i] = p_lo[i]
        else:
            # AA 角:张开基准取中位数(相对它取带符号偏差)
            m.aa_open[i] = np.median(mp_angles[:, i])

    # 防止 open≈close 造成除零/过敏感
    span = m.open_deg - m.close_deg
    too_narrow = (np.abs(span) < 8.0) & IS_FE
    m.close_deg[too_narrow] = m.open_deg[too_narrow] - 15.0

    print("FE 锚点标定完成(数据分位数)。各关节 open/close(度):")
    for i, n in enumerate(JOINT_NAMES):
        if IS_FE[i]:
            print(f"  {n:16s} open={m.open_deg[i]:6.1f} close={m.close_deg[i]:6.1f}")

    # ---- 可选:用骨段方向残差轻量精修 AA 缩放(方向对尺度/朝向不敏感,最稳) ----
    if os.path.exists(_CALIB_LANDMARKS):
        step = max(1, M // 300)
        idx = np.arange(0, M, step)
        reference_landmarks = landmarks_from_angles(np.zeros(NUM_JOINTS))
        tgt_dirs = np.array([
            _finger_dir_vectors_mp(mp_lms[j], reference_landmarks) for j in idx
        ])  # (K,5,3), all in the UmeTrack palm frame

        def residuals(aa_scale_scalar):
            m.aa_scale[:] = aa_scale_scalar[0]
            res = []
            for jj, j in enumerate(idx):
                u = m.map(mp_angles[j])
                res.append((_finger_dir_vectors_ume(u) - tgt_dirs[jj]).reshape(-1))
            return np.concatenate(res)

        r0 = residuals([m.aa_scale[0]])
        print(f"AA 精修前 方向残差 RMS = {np.sqrt(np.mean(r0**2)):.4f}")
        sol = least_squares(residuals, [1.0], bounds=([0.2], [3.0]),
                            method="trf", max_nfev=20, verbose=1)
        r1 = residuals(sol.x)
        print(f"AA 精修后 方向残差 RMS = {np.sqrt(np.mean(r1**2)):.4f}  aa_scale={sol.x[0]:.3f}")

    m.save()
    print("已保存标定参数到 angle_map_params.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="采集标定数据")
    ap.add_argument("--fit", action="store_true", help="拟合并保存参数")
    args = ap.parse_args()
    if not (args.collect or args.fit):
        ap.print_help()
    if args.collect:
        collect()
    if args.fit:
        fit()
