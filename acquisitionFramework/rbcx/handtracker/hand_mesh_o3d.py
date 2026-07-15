"""基于 Open3D 的 UmeTrack LBS 蒙皮手实时渲染器。

替换旧的 hand3d_o3d.Hand3DVisualizer(球+圆柱胶囊骨架)。本渲染器渲染的是
一只真正的连续表面蒙皮手(788 顶点 / 1544 三角面),由 20 个关节角经 UmeTrack
前向运动学 + 线性混合蒙皮(LBS)驱动,观感接近真实手。

关键设计:
- 拓扑固定:三角面索引一次性设定,每帧只更新顶点 + 重算法线,零重建、无累积误差。
- 驱动解耦:update() 接收 20 个“已映射到 UmeTrack 语义”的关节角(弧度),
  与驱动来源(MediaPipe 视觉 / EMG 预测)无关。
- 时间平滑:对关节角做指数滑动平均(EMA),消除逐帧抖动,视觉更流畅。
- 全局朝向:可选 wrist_transform(4x4),让手随真实手腕朝向转动;None 时固定朝向。
- 非阻塞:poll() 内部 poll_events()+update_renderer(),嵌入主循环,与 OpenCV 窗共存。

接口与旧 Hand3DVisualizer 对齐(update/poll/close),但 update 语义从“21 landmarks”
改为“20 关节角”,便于 main_o3d.py 平滑替换。
"""
import numpy as np

from rbcx.handtracker.open3d_import import import_open3d_desktop

try:
    o3d = import_open3d_desktop()
except ImportError as e:
    raise ImportError("需要安装 open3d: pip install open3d") from e

from rbcx.handmodel import mesh_from_angles, mesh_triangles, load_hand_model


# 手部材质颜色。emg2pose 官方 overview 图用灰白哑光;这里取略暖的浅灰,接近官方观感。
SKIN_COLOR = [0.72, 0.72, 0.74]


class HandMeshVisualizer:
    def __init__(self, window_name="UmeTrack Hand (LBS skinned mesh)",
                 width=760, height=760, smoothing=0.5, subdivide=0, fixed_wrist=True):
        """
        smoothing : 关节角 EMA 平滑系数 alpha(0=不平滑,越大越平滑越滞后)。0.5 折中。
        subdivide : Loop 细分迭代次数。0=原始低模(棱角明显);1=柔和(788->3119 顶点,
                    推荐,接近 emg2pose 论文观感);2=更光滑但更慢。
        fixed_wrist: True=手腕朝向固定(不随真实手翻滚,与图2一致);
                     False=用外部传入的 wrist_transform 驱动全局朝向。
        """
        self.window_name = window_name
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.subdivide = int(subdivide)
        self.fixed_wrist = bool(fixed_wrist)
        self._angles_ema = None      # 平滑后的 20 关节角(弧度)
        self._last_angles = np.zeros(20, dtype=np.float32)
        self._last_timestamp = None
        self._view_fitted = False

        load_hand_model()
        self._tris = mesh_triangles().astype(np.int32)

        # 基座旋转:UmeTrack rest pose 手指沿 +X(水平),这里旋成手指朝屏幕上方(竖直竖立)。
        # 绕 Z 轴 +90°: +X -> +Y。纯显示层旋转,不影响 FK 关节角语义。
        self._base_R = np.array([[0.0, -1.0, 0.0],
                                 [1.0,  0.0, 0.0],
                                 [0.0,  0.0, 1.0]], dtype=np.float64)

        # 初始 rest pose 网格(经细分)
        verts0, _ = mesh_from_angles(np.zeros(20, dtype=np.float32))
        self.mesh = self._build_mesh(verts0)

        # 用带按键回调的可视化器,便于主循环注册“切换驱动源”等快捷键
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_name, width=width, height=height)
        self.vis.add_geometry(self.mesh)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.11, 0.11, 0.13])
        opt.light_on = True
        opt.mesh_show_back_face = True
        # 柔和观感:关闭高光,平滑着色(法线插值),接近黏土/皮肤质感
        try:
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
            opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        except Exception:
            pass

        self._set_view()

    def _build_mesh(self, verts):
        """由蒙皮顶点(788,3)构建(可选细分的)平滑着色三角网格。应用基座旋转使手竖立。"""
        verts = np.asarray(verts, dtype=np.float64) @ self._base_R.T
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(verts)
        m.triangles = o3d.utility.Vector3iVector(self._tris)
        if self.subdivide > 0:
            m = m.subdivide_loop(number_of_iterations=self.subdivide)
        m.paint_uniform_color(SKIN_COLOR)
        m.compute_vertex_normals()
        return m

    def _set_view(self):
        """相机:正对手背看(五指铺开可见)。

        基座旋转后:手指方向≈+Y(朝上),掌横向(食指->小指)≈-Z,掌面法向≈X。
        因此看手背要让相机 front 沿掌法向(-X),up 沿手指方向(+Y);
        这样五指在画面里左右铺开,与 emg2pose 官方渲染一致(而非从侧面看成重叠)。
        """
        try:
            vc = self.vis.get_view_control()
            vc.set_front([-1.0, 0.0, 0.0])   # 沿掌面法向正视手背
            vc.set_up([0.0, 1.0, 0.0])       # 手指朝上
            vc.set_lookat([14.0, 93.0, -2.0])  # 旋转后手中心(实测均值)
            vc.set_zoom(0.8)
        except Exception:
            pass

    def _accept_timestamp(self, timestamp):
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return False
        self._last_timestamp = timestamp
        return True

    def update(self, joint_angles_20, wrist_transform=None, timestamp=None):
        """用 20 个 UmeTrack 语义关节角(弧度)更新蒙皮手。angles 为 None 时保持上一帧。

        wrist_transform: 可选 (4,4) 全局手腕朝向仿射。
        """
        if joint_angles_20 is None:
            return False
        if timestamp is None:
            import time
            timestamp = time.monotonic()
        previous_timestamp = self._last_timestamp
        if not self._accept_timestamp(timestamp):
            return False
        ja = np.asarray(joint_angles_20, dtype=np.float32).reshape(-1)[:20]
        if ja.shape[0] != 20 or not np.all(np.isfinite(ja)):
            return False

        # 指数平滑
        if self._angles_ema is None:
            self._angles_ema = ja.copy()
        else:
            dt = max(float(timestamp) - float(previous_timestamp), 1e-4)
            a = self.smoothing ** (dt * 60.0)
            self._angles_ema = a * self._angles_ema + (1.0 - a) * ja
        self._last_angles = self._angles_ema

        # 固定手腕朝向时忽略外部 wrist_transform(手不随真实手翻滚,与图2一致)
        wxf = None if self.fixed_wrist else wrist_transform

        verts, _ = mesh_from_angles(self._angles_ema, wrist_transform=wxf)
        if self.subdivide == 0:
            transformed = np.asarray(verts, dtype=np.float64) @ self._base_R.T
            self.mesh.vertices = o3d.utility.Vector3dVector(transformed)
            self.mesh.compute_vertex_normals()
        else:
            new_mesh = self._build_mesh(verts)
            self.mesh.vertices = new_mesh.vertices
            self.mesh.vertex_normals = new_mesh.vertex_normals
        self.vis.update_geometry(self.mesh)

        if not self._view_fitted:
            self.vis.reset_view_point(True)
            self._set_view()
            self._view_fitted = True
        return True

    def register_key(self, key: str, callback):
        """注册按键回调(如 'V'/'E' 切换驱动源)。callback 签名 fn(vis)->bool。"""
        try:
            self.vis.register_key_callback(ord(key.upper()), callback)
        except Exception as e:
            print(f"[HandMeshVisualizer] 注册按键 {key} 失败: {e}")

    def poll(self):
        """非阻塞刷新一帧。返回窗口是否仍打开。"""
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass
