"""采集入口:用 UmeTrack LBS 蒙皮手(Open3D 实时渲染)替换旧的胶囊骨架手。

与旧版(球+圆柱骨架)的区别:
- 3D 反馈窗渲染的是真正的连续表面蒙皮手(788 顶点),由 20 个关节角经
  UmeTrack 前向运动学 + 线性混合蒙皮驱动,观感接近真实手(见 rbcx/handmodel)。
- 驱动源解耦且可实时切换:
    * vision 源:MediaPipe 视觉算出的 20 关节角
    * emg    源:EMG 回归模型预测的 20 关节角(EMG_regressor)
  两条路径共用同一角度映射层(mediapipe 几何角 -> UmeTrack 弧度)与同一 FK/蒙皮/渲染。
- 保留 MediaPipe 实拍手骨架 + RGB 预览窗;EMG 采集/训练/保存逻辑完全不变。

快捷键(焦点在 3D 窗口时):
    V -> 切到 vision(视觉关节角驱动)
    E -> 切到 emg(EMG 预测关节角驱动)
退出:关闭 3D 窗口,或在 RGB 预览窗按 ESC/q,或控制台 Ctrl+C。
"""
import os
import sys
import time

# 抑制 TensorFlow 冷启动的无关日志(mediapipe 会连锁拉起 TF)。须在任何 TF/mediapipe 导入前设置。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# 启动即时反馈:import mediapipe 会连锁加载 TensorFlow,首次约需 10~15 秒,
# 期间进程看似“无响应”。这里先明确提示,避免用户误以为卡死而 Ctrl+C 中断
# (那会在 import 半途抛出 KeyboardInterrupt,产生一长串吓人的 traceback)。
print("正在加载依赖库(mediapipe / tensorflow 首次导入约需 10~15 秒,请稍候,勿按 Ctrl+C)...",
      flush=True)
_t_import = time.time()

import numpy as np

from rbcx.handtracker.mediapipe import MediaPipeHandTracker
from rbcx.handtracker.hand_mesh_o3d import HandMeshVisualizer
from rbcx.handmodel.angle_map import AngleMapper
from EMG_regression import EMG_regressor
import config

print(f"依赖库加载完成(耗时 {time.time() - _t_import:.1f}s)。", flush=True)


# 驱动源状态(用可变容器以便 Open3D 按键回调闭包修改)
_source = {"mode": "vision"}


def _get_underlying_tracker(hand_tracker):
    """无论 EMG 模式与否,取到底层 MediaPipeHandTracker(用于拿视觉角/landmarks)。"""
    if isinstance(hand_tracker, EMG_regressor):
        return hand_tracker.handTracker
    return hand_tracker


def _wrist_transform_from_landmarks(lm):
    """从 MediaPipe world landmarks(21,3) 估计全局手腕朝向的 4x4 仿射。

    构造正交基:纵轴 = wrist->middle_mcp,掌面法向 = (wrist->index_mcp) x (wrist->pinky_mcp),
    第三轴 = 两者叉乘。把 UmeTrack rest pose(手指沿 +X)对齐到该朝向。
    landmarks 不可用时返回 None(渲染器将用固定朝向)。
    """
    if lm is None:
        return None
    lm = np.asarray(lm, dtype=np.float64)
    if lm.shape != (21, 3):
        return None
    wrist = lm[0]
    idx_mcp, mid_mcp, pinky_mcp = lm[5], lm[9], lm[17]

    x_axis = mid_mcp - wrist                      # 手指延伸方向(对齐 UmeTrack +X)
    nx = np.linalg.norm(x_axis)
    if nx < 1e-6:
        return None
    x_axis /= nx
    normal = np.cross(idx_mcp - wrist, pinky_mcp - wrist)  # 掌面法向
    nn = np.linalg.norm(normal)
    if nn < 1e-6:
        return None
    z_axis = normal / nn
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-9)
    z_axis = np.cross(x_axis, y_axis)             # 重正交化

    R = np.eye(4, dtype=np.float32)
    R[:3, 0] = x_axis
    R[:3, 1] = y_axis
    R[:3, 2] = z_axis
    return R


def main():
    # target_fps=60 + MJPG:让关节角采样率跟上 60Hz 摄像头,实时反馈更顺滑。
    # 若 MediaPipe 在 60fps 下跟不上(实际更新率打印在控制台),可把 model_complexity 降到 0。
    hand_tracker_mediapipe = MediaPipeHandTracker(
        camera_index=0, mirror=False, show_window=True,
        use_2D_coord_for_angles=True,
        target_fps=60, frame_width=640, frame_height=480, use_mjpg=True,
    )

    if config.EMG:
        hand_tracker = EMG_regressor(hand_tracker_mediapipe)
    else:
        hand_tracker = hand_tracker_mediapipe
    hand_tracker.start()

    underlying = _get_underlying_tracker(hand_tracker)
    mapper = AngleMapper()
    if not mapper.calibrated:
        print("[提示] 未找到角度标定参数,使用解析初值。可运行 calibrate_angle_map.py 精修。")

    # subdivide=1 让蒙皮表面柔和(接近图2观感);fixed_wrist=True 固定手腕朝向不翻滚
    hand3d = HandMeshVisualizer(smoothing=0.5, subdivide=1, fixed_wrist=True)

    # 注册驱动源切换快捷键
    def _to_vision(vis):
        _source["mode"] = "vision"
        print(">> 驱动源 = vision(视觉关节角)")
        return False
    def _to_emg(vis):
        _source["mode"] = "emg"
        print(">> 驱动源 = emg(EMG 预测关节角)")
        return False
    hand3d.register_key("V", _to_vision)
    hand3d.register_key("E", _to_emg)

    print("Started (UmeTrack 蒙皮手 + RGB 预览)。3D 窗口按 V/E 切换驱动源;ESC/q/Ctrl+C 退出。")
    if not config.EMG:
        print("[提示] config.EMG=False,emg 源不可用,仅 vision 源有效。")

    try:
        while True:
            # 推进采集节拍。EMG 模式下 get_hand_state 内部会同步采 EMG+记录视觉角作 label,
            # 并返回 EMG 预测角;纯 MediaPipe 模式返回视觉角。
            hand_state = hand_tracker.get_hand_state("Right")

            # 选择驱动用的 20 个 mediapipe 语义角
            if _source["mode"] == "emg" and config.EMG:
                mp_angles = np.asarray(hand_state["angles_list"], dtype=np.float64)
            else:
                mp_angles = np.asarray(underlying.get_mediapipe_angles("Right"), dtype=np.float64)

            # 映射到 UmeTrack 弧度语义
            u_angles = mapper.map(mp_angles)

            # 全局手腕朝向:仅在非固定朝向模式下才用视觉 landmarks 估计(省开销)
            wrist_xf = None
            if not hand3d.fixed_wrist:
                state = underlying.get_hand_state("Right")
                wrist_xf = _wrist_transform_from_landmarks(state.get("landmarks", None))

            hand3d.update(u_angles, wrist_transform=wrist_xf)
            if not hand3d.poll():
                break

            if getattr(hand_tracker, "show_window", False):
                if not hand_tracker.poll_gui():
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        hand_tracker.stop()
        hand3d.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断,退出。")
        sys.exit(0)
