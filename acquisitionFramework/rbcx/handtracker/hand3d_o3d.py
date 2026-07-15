"""
基于 Open3D 的五指手实时 3D 可视化(圆柱包皮实体手)。

用 MediaPipe 输出的 21 个 3D 关键点(world 坐标)渲染一只有体积感的实体手:
- 每根骨段(HAND_CONNECTIONS)包一根圆柱(cylinder)
- 每个关节包一个球(sphere)
组成"胶囊串"式实体手,比纯骨架更接近真实手观感,且完全实时。

设计要点:
- 非阻塞刷新:poll() 内部调 poll_events()+update_renderer(),嵌入主循环,不与 OpenCV 预览窗冲突。
- 每帧对每根圆柱重算 仿射变换(把单位圆柱从沿 +Z、单位长 变换到 两端点之间):
  旋转(+Z -> 骨段方向) + 缩放(长度) + 平移(中点)。用保存的"基准顶点"做变换后写回,避免累积误差。
- 坐标变换:MediaPipe world(Y 向下、Z 朝屏内)翻转成 Open3D 习惯(Y 向上),并放大尺度。
- landmarks 为 None(镜头前无手)时保持上一帧,不崩溃。
"""
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    raise ImportError("需要安装 open3d: pip install open3d") from e


# 21 点手部骨架连线(含五指 + 掌根),与 MediaPipe 标准一致
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),         # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),    # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20), # 小指
    (0, 17),                                # 掌根连线
]

FINGER_POINTS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
FINGER_COLORS = {
    "thumb":  [0.95, 0.55, 0.45],
    "index":  [0.95, 0.80, 0.45],
    "middle": [0.55, 0.90, 0.55],
    "ring":   [0.45, 0.75, 0.95],
    "pinky":  [0.80, 0.55, 0.92],
}
SKIN_COLOR = [0.92, 0.72, 0.62]    # 指骨圆柱:肉色
JOINT_COLOR = [0.85, 0.55, 0.48]   # 关节球:略深肉色
WRIST_COLOR = [0.75, 0.60, 0.55]


def _point_colors():
    colors = np.tile(JOINT_COLOR, (21, 1)).astype(np.float64)
    colors[0] = WRIST_COLOR
    for finger, ids in FINGER_POINTS.items():
        for idx in ids:
            colors[idx] = FINGER_COLORS[finger]
    return colors


def _transform_landmarks(lm):
    """
    MediaPipe world 坐标 -> Open3D 观察坐标。
    world 单位是米(整手展布约 0.15-0.20m,单指骨约 0.02-0.04m),放大到便于观察的尺度。
    放大 100 倍后:整手约 15-20 单位,单指骨约 2-4 单位 —— 与球半径(0.5)、圆柱半径(0.35)成合理比例。
    """
    pts = np.asarray(lm, dtype=np.float64).copy()
    pts[:, 1] *= -1.0
    pts[:, 2] *= -1.0
    pts *= 100.0
    return pts


def _rotation_from_z(direction, eps=1e-9):
    """返回把 +Z 轴旋到 direction 方向的 3x3 旋转矩阵(罗德里格斯公式)。"""
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < eps:
        return np.eye(3)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    s = np.linalg.norm(v)
    if s < eps:
        # 平行或反平行
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


class Hand3DVisualizer:
    def __init__(self, window_name="Hand 3D (5 fingers, solid)", width=760, height=760,
                 joint_radius=0.55, bone_radius=0.38):
        self.window_name = window_name
        self.joint_radius = joint_radius
        self.bone_radius = bone_radius
        self._point_colors = _point_colors()
        self._last_pts = None
        self._view_fitted = False  # 首次拿到真实手数据后自动适配视角

        init_pts = self._default_open_hand()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)

        # ---- 关节球 ----
        # 保存每个球的"基准顶点"(以原点为中心的单位球*半径),每帧平移到关键点位置
        self.spheres = []
        self._sphere_base = []
        for i in range(21):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=self.joint_radius, resolution=12)
            s.compute_vertex_normals()
            s.paint_uniform_color(self._point_colors[i])
            base_v = np.asarray(s.vertices).copy()  # 原点为中心
            s.translate(init_pts[i])
            self.spheres.append(s)
            self._sphere_base.append(base_v)
            self.vis.add_geometry(s)

        # ---- 骨段圆柱 ----
        # 基准圆柱:沿 +Z、单位高、以原点为中心。每帧做 旋转*缩放 后平移到骨段中点。
        self.cylinders = []
        self._cyl_base = []
        self._cyl_colors = []
        for (a, b) in HAND_CONNECTIONS:
            cyl = o3d.geometry.TriangleMesh.create_cylinder(
                radius=self.bone_radius, height=1.0, resolution=12, split=1)
            cyl.compute_vertex_normals()
            col = (np.array(self._point_colors[a]) + np.array(self._point_colors[b])) / 2.0
            col = 0.5 * col + 0.5 * np.array(SKIN_COLOR)
            cyl.paint_uniform_color(col)
            self.cylinders.append(cyl)
            self._cyl_base.append(np.asarray(cyl.vertices).copy())
            self._cyl_colors.append(col)
            self.vis.add_geometry(cyl)

        # 应用初始姿态
        self._apply_pose(init_pts)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.11, 0.11, 0.13])
        opt.light_on = True

        self._set_view()
        self._last_pts = init_pts

    @staticmethod
    def _default_open_hand():
        """张开手的占位 21 点(尺度与 _transform_landmarks 放大后一致,整手约 15 单位)。"""
        pts = np.zeros((21, 3))
        pts[0] = [0, -6, 0]  # 手腕
        # 五指根部 x 展开,每指 4 节向上延伸(每节约 2.5 单位)
        base_x = {1: -5.5, 5: -3.0, 9: -0.5, 13: 2.0, 17: 4.5}
        for base, bx in base_x.items():
            for k in range(4):
                pts[base + k] = [bx + k * 0.15, -3.0 + k * 2.5, 0]
        return pts

    def _set_view(self):
        try:
            vc = self.vis.get_view_control()
            vc.set_front([0.0, 0.0, -1.0])
            vc.set_up([0.0, 1.0, 0.0])
            vc.set_lookat([0.0, 0.0, 0.0])
            vc.set_zoom(0.8)
        except Exception:
            pass

    def _apply_pose(self, pts):
        """把所有球和圆柱更新到 pts(21,3) 指定的姿态。"""
        # 关节球:基准顶点 + 目标中心
        for i in range(21):
            new_v = self._sphere_base[i] + pts[i]
            self.spheres[i].vertices = o3d.utility.Vector3dVector(new_v)
            self.spheres[i].compute_vertex_normals()
            self.vis.update_geometry(self.spheres[i])

        # 骨段圆柱:基准(沿+Z单位高) -> 旋转到骨段方向 * 缩放到骨段长 -> 平移到中点
        for ci, (a, b) in enumerate(HAND_CONNECTIONS):
            pa, pb = pts[a], pts[b]
            seg = pb - pa
            length = float(np.linalg.norm(seg))
            if length < 1e-6:
                continue
            R = _rotation_from_z(seg)
            base = self._cyl_base[ci].copy()
            base[:, 2] *= length                 # 高度缩放(基准高=1)
            new_v = base @ R.T + (pa + pb) / 2.0  # 旋转 + 平移到中点
            self.cylinders[ci].vertices = o3d.utility.Vector3dVector(new_v)
            self.cylinders[ci].compute_vertex_normals()
            self.vis.update_geometry(self.cylinders[ci])

    def update(self, landmarks):
        """用新的 21x3 world 坐标更新实体手。landmarks 为 None 时保持上一帧。"""
        if landmarks is None:
            return
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.shape != (21, 3):
            return
        pts = _transform_landmarks(lm)
        self._apply_pose(pts)
        self._last_pts = pts
        # 首次拿到真实手数据时,把相机重置到能看清整只手的视角
        if not self._view_fitted:
            self.vis.reset_view_point(True)
            self._set_view()
            self._view_fitted = True

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
