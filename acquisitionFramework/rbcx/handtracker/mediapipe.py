# mediapipe_handtracker.py

import time
import threading
import platform

from copy import deepcopy

import cv2
import numpy as np
import mediapipe as mp


class MediaPipeHandTracker:
    """
    Threaded MediaPipe Hands, OpenCV overlay and 3D joint angles estimation and display
    Call start() to launch the capture/processing thread
    Optionally shows one OpenCV window (set show_window=True/False)
    Get angles anytime with get_mediapipe_angles()
    Call stop() to stop the thread

    Parameters
    ----------
    camera_index : int: OpenCV camera index
    mirror : bool: if True, mirror the display (processing remains non-mirrored)
    show_window : bool: if True, show a live window from the internal thread. 
                        you can also access the image simply with 'get_last_frame()' without showing the window
    MediaPipe Hands options: max_num_hands, model_complexity (1 for best model, 0 for faster), min_detection_confidence, min_tracking_confidence
    """

    def __init__(
        self,
        camera_index: int = 0,
        window_name: str = "Hand Tracking",
        mirror: bool = False,
        show_window: bool = True,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_2D_coord_for_angles=True,
        capture_backend=None,
        target_fps: int = 60,
        frame_width: int = 640,
        frame_height: int = 480,
        use_mjpg: bool = True,
    ):
        self.camera_index = camera_index
        self.window_name = window_name
        self.mirror = mirror
        self.show_window = show_window

        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # 采集帧率/编码/分辨率:关节角采样率上限直接由摄像头实际输出帧率决定。
        # 很多 USB 摄像头需 MJPG 编码才能达到 60fps(YUY2 常被限到 30)。
        self.target_fps = int(target_fps)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.use_mjpg = bool(use_mjpg)

        # Windows 上 MSMF 后端设 fps/编码经常不生效,DSHOW 更可靠;macOS 用 AVFoundation。
        if capture_backend is not None:
            self.capture_backend = capture_backend
        elif platform.system() == "Darwin":
            self.capture_backend = cv2.CAP_AVFOUNDATION
        elif platform.system() == "Windows":
            self.capture_backend = cv2.CAP_DSHOW
        else:
            self.capture_backend = cv2.CAP_ANY

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._latest_right_angles = {
            'timestamp': time.time(),
            'angles': {
                'THUMB_CMC_FE': 0, 'THUMB_CMC_AA': 0, 'THUMB_MCP_FE': 0, 'THUMB_IP_FE': 0,
                'INDEX_MCP_AA': 0, 'INDEX_MCP_FE': 0, 'INDEX_PIP_FE': 0, 'INDEX_DIP_FE': 0,
                'MIDDLE_MCP_AA': 0, 'MIDDLE_MCP_FE': 0, 'MIDDLE_PIP_FE': 0, 'MIDDLE_DIP_FE': 0,
                'RING_MCP_AA': 0, 'RING_MCP_FE': 0, 'RING_PIP_FE': 0, 'RING_DIP_FE': 0,
                'PINKY_MCP_AA': 0, 'PINKY_MCP_FE': 0, 'PINKY_PIP_FE': 0, 'PINKY_DIP_FE': 0,
            },
            'landmarks': None,          # np.ndarray of shape (21, 3)
            'landmarks_type': None,     # 'world' or 'normalized'
            'normalized_landmarks': None,
            'world_landmarks': None,
            'handedness': 'Right',
        }
        self._latest_left_angles = {
            'timestamp': time.time(),
            'angles': {
                'THUMB_CMC_FE': 0, 'THUMB_CMC_AA': 0, 'THUMB_MCP_FE': 0, 'THUMB_IP_FE': 0,
                'INDEX_MCP_AA': 0, 'INDEX_MCP_FE': 0, 'INDEX_PIP_FE': 0, 'INDEX_DIP_FE': 0,
                'MIDDLE_MCP_AA': 0, 'MIDDLE_MCP_FE': 0, 'MIDDLE_PIP_FE': 0, 'MIDDLE_DIP_FE': 0,
                'RING_MCP_AA': 0, 'RING_MCP_FE': 0, 'RING_PIP_FE': 0, 'RING_DIP_FE': 0,
                'PINKY_MCP_AA': 0, 'PINKY_MCP_FE': 0, 'PINKY_PIP_FE': 0, 'PINKY_DIP_FE': 0,
            },
            'landmarks': None,          # np.ndarray of shape (21, 3)
            'landmarks_type': None,     # 'world' or 'normalized'
            'normalized_landmarks': None,
            'world_landmarks': None,
            'handedness': 'Left',
        }

        # 20 个关节角,严格按 emg2pose constants.py 的顺序(FE=屈曲, AA=外展/内收)
        self.joint_names = [
            'THUMB_CMC_FE', 'THUMB_CMC_AA', 'THUMB_MCP_FE', 'THUMB_IP_FE',
            'INDEX_MCP_AA', 'INDEX_MCP_FE', 'INDEX_PIP_FE', 'INDEX_DIP_FE',
            'MIDDLE_MCP_AA', 'MIDDLE_MCP_FE', 'MIDDLE_PIP_FE', 'MIDDLE_DIP_FE',
            'RING_MCP_AA', 'RING_MCP_FE', 'RING_PIP_FE', 'RING_DIP_FE',
            'PINKY_MCP_AA', 'PINKY_MCP_FE', 'PINKY_PIP_FE', 'PINKY_DIP_FE',
        ]

        self._latest_frame = None
        self._running = False
        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.pose_fps = 0.0
        self.processing_fps = 0.0   # 向后兼容:等于 inference_fps
        self.use_2D_coord_for_angles = use_2D_coord_for_angles

        # New: OpenCV GUI must be managed on the main thread
        self._window_created = False
        self._thread_exception = None

    def start(self):
        """Start the background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread_exception = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="MediaPipeHandTracker"
        )
        self._thread.start()
        self._running = True

    def stop(self):
        """Signal the worker thread to stop. Call this from the main thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False
        self.close_window()

    def close_window(self):
        """Destroy the OpenCV window from the main thread."""
        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self._window_created = False

    def is_running(self) -> bool:
        return self._running and not self._stop.is_set()
    
    def _raise_if_thread_failed(self):
        with self._lock:
            exc = self._thread_exception
        if exc is not None:
            raise RuntimeError("MediaPipeHandTracker worker thread crashed") from exc


    def get_mediapipe_angles(self, handeness="Right"):
        """
        Return the most recent set of angles.

        Order:
        ['Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Index_MCP', 'Index_PIP', 'Index_DIP',
        'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Ring_MCP', 'Ring_PIP', 'Ring_DIP',
        'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']
        """
        self._raise_if_thread_failed()

        with self._lock:
            if handeness == "Right":
                mediapipe_angles = deepcopy(self._latest_right_angles['angles'])
            else:
                mediapipe_angles = deepcopy(self._latest_left_angles['angles'])

        return [mediapipe_angles[k] for k in self.joint_names]
    
    def get_hand_state(self, handeness="Right"):
        """
        Return latest angles + landmarks from the same frame.

        Output dict:
            {
                'timestamp': float,
                'angles_list': [20 floats],
                'angles_dict': {...},
                'landmarks': np.ndarray shape (21, 3) or None,
                'landmarks_type': 'world' | 'normalized' | None
            }
        """
        self._raise_if_thread_failed()

        with self._lock:
            latest = self._latest_right_angles if handeness == "Right" else self._latest_left_angles

            timestamp = latest['timestamp']
            angles_dict = deepcopy(latest['angles'])
            landmarks = None if latest['landmarks'] is None else deepcopy(latest['landmarks'])
            landmarks_type = latest.get('landmarks_type', None)
            normalized_landmarks = (
                None if latest.get('normalized_landmarks') is None
                else deepcopy(latest['normalized_landmarks'])
            )
            world_landmarks = (
                None if latest.get('world_landmarks') is None
                else deepcopy(latest['world_landmarks'])
            )
            handedness = latest.get('handedness', handeness)

        return {
            'timestamp': timestamp,
            'angles_list': [angles_dict[k] for k in self.joint_names],
            'angles_dict': angles_dict,
            'landmarks': landmarks,
            'landmarks_type': landmarks_type,
            'normalized_landmarks': normalized_landmarks,
            'world_landmarks': world_landmarks,
            'handedness': handedness,
        }

    def _select_angle_points(self, normalized_landmarks, world_landmarks):
        if self.use_2D_coord_for_angles or world_landmarks is None:
            return normalized_landmarks
        return world_landmarks

    def _publish_observation(
        self,
        handedness,
        timestamp,
        angles,
        normalized_landmarks,
        world_landmarks,
    ):
        preferred = world_landmarks if world_landmarks is not None else normalized_landmarks
        payload = {
            'timestamp': float(timestamp),
            'angles': deepcopy(angles),
            'landmarks': None if preferred is None else deepcopy(preferred),
            'landmarks_type': 'world' if world_landmarks is not None else 'normalized',
            'normalized_landmarks': (
                None if normalized_landmarks is None else deepcopy(normalized_landmarks)
            ),
            'world_landmarks': None if world_landmarks is None else deepcopy(world_landmarks),
            'handedness': handedness,
        }
        with self._lock:
            if handedness == "Right":
                self._latest_right_angles = payload
            elif handedness == "Left":
                self._latest_left_angles = payload
        

    def get_last_frame(self):
        """Return a copy of the last annotated BGR frame (or None)."""
        self._raise_if_thread_failed()

        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def poll_gui(self) -> bool:
        """
        Show the latest frame from the MAIN thread only.

        Returns False when the user presses ESC or q.
        """
        self._raise_if_thread_failed()

        if not self.show_window:
            return True

        frame = self.get_last_frame()
        if frame is None:
            return True

        try:
            if not self._window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._window_created = True

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self.stop()
                return False
            return True
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV GUI failed. On macOS, cv2.namedWindow/imshow/waitKey must run on the main thread."
            ) from exc

    def _run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index, self.capture_backend)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera index {self.camera_index} with backend {self.capture_backend}"
                )

            # 提高采集帧率:顺序很关键,须先设编码(FOURCC)再设分辨率/帧率,否则常不生效。
            if self.use_mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            # 减小驱动缓冲,降低延迟(部分后端支持)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"[MediaPipeHandTracker] 摄像头实际: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.0f}fps "
                  f"(目标 {self.target_fps}fps, MJPG={self.use_mjpg})")

            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as hands:

                # IMPORTANT:
                # No cv2.namedWindow / cv2.imshow / cv2.waitKey here.
                # This worker thread only captures/processes frames.

                # 分开统计采集、推理和成功右手姿态率。
                _fps_t0 = time.time()
                _capture_n = 0
                _inference_n = 0
                _pose_n = 0
                self.processing_fps = 0.0

                while not self._stop.is_set():
                    ok, frame_bgr = cap.read()
                    if not ok:
                        time.sleep(0.005)
                        continue

                    _capture_n += 1
                    _elapsed = time.time() - _fps_t0
                    if _elapsed >= 2.0:
                        self.capture_fps = _capture_n / _elapsed
                        self.inference_fps = _inference_n / _elapsed
                        self.pose_fps = _pose_n / _elapsed
                        self.processing_fps = self.inference_fps
                        print(
                            "[MediaPipeHandTracker] "
                            f"capture={self.capture_fps:.1f} inference={self.inference_fps:.1f} "
                            f"right_pose={self.pose_fps:.1f} fps"
                        )
                        _fps_t0 = time.time()
                        _capture_n = _inference_n = _pose_n = 0

                    # Processing is always done on the non-mirrored frame
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = hands.process(rgb)
                    _inference_n += 1
                    rgb.flags.writeable = True

                    annotated = frame_bgr.copy() if self.show_window else None
                    h, w = frame_bgr.shape[:2]
                    frame_timestamp = time.time()

                    hands_out = []
                    if results.multi_hand_landmarks:
                        for i, hand_lms in enumerate(results.multi_hand_landmarks):
                            # Draw skeleton
                            if annotated is not None:
                                mp_draw.draw_landmarks(
                                    annotated,
                                    hand_lms,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_styles.get_default_hand_landmarks_style(),
                                    mp_styles.get_default_hand_connections_style(),
                                )

                            # Normalized image landmarks (always available)
                            normalized_pts = np.array(
                                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                                dtype=np.float64
                            )

                            # True world landmarks when available
                            if (
                                results.multi_hand_world_landmarks
                                and len(results.multi_hand_world_landmarks) > i
                            ):
                                world_pts = np.array(
                                    [[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[i].landmark],
                                    dtype=np.float64
                                )
                            else:
                                world_pts = None

                            # Keep the choice for angle computation
                            angle_pts = self._select_angle_points(normalized_pts, world_pts)

                            # For thumb pinch logic, prefer true world landmarks
                            stored_landmarks = world_pts if world_pts is not None else normalized_pts
                            stored_landmarks_type = 'world' if world_pts is not None else 'normalized'

                            # Angles from vectors
                            angles = self._compute_joint_angles(angle_pts, mp_hands)

                            # 2D positions for labels
                            pts2d = [(int(p.x * w), int(p.y * h)) for p in hand_lms.landmark]
                            if annotated is not None:
                                self._draw_angle_labels(annotated, pts2d, angles, mp_hands)

                            # Handedness (swap because processing image is not mirrored)
                            handed = None
                            if results.multi_handedness and len(results.multi_handedness) > i:
                                label = results.multi_handedness[i].classification[0].label
                                handed = ('Left' if label == 'Right' else 'Right')

                                if annotated is not None:
                                    wx, wy = pts2d[mp_hands.HandLandmark.WRIST.value]
                                    cv2.putText(
                                        annotated, handed, (wx - 20, wy - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                                        2, cv2.LINE_AA,
                                    )

                            hands_out.append({
                                'handedness': handed,
                                'angles': angles,
                                'normalized_landmarks': normalized_pts,
                                'world_landmarks': world_pts,
                            })

                    for hand in hands_out:
                        self._publish_observation(
                            hand['handedness'], frame_timestamp, hand['angles'],
                            hand['normalized_landmarks'], hand['world_landmarks'],
                        )
                        if hand['handedness'] == "Right":
                            _pose_n += 1

                    if annotated is not None:
                        display = cv2.flip(annotated, 1) if self.mirror else annotated
                        with self._lock:
                            self._latest_frame = display

        except Exception as exc:
            with self._lock:
                self._thread_exception = exc
            self._stop.set()

        finally:
            if cap is not None:
                cap.release()
            self._running = False


    @staticmethod
    def _angle_between(v1, v2, eps=1e-9):
        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)
        n1 = np.linalg.norm(v1) + eps
        n2 = np.linalg.norm(v2) + eps
        c = np.dot(v1, v2) / (n1 * n2)
        c = np.clip(c, -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    @classmethod
    def _angle_3pts(cls, a, b, c):
        # angle at b defined by vectors (a-b) and (c-b)
        return cls._angle_between(np.array(a) - np.array(b), np.array(c) - np.array(b))

    @staticmethod
    def _palm_center(world_pts, mp_hands):
        idx = [
            mp_hands.HandLandmark.WRIST.value,
            mp_hands.HandLandmark.THUMB_CMC.value,
            mp_hands.HandLandmark.INDEX_FINGER_MCP.value,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value,
            mp_hands.HandLandmark.RING_FINGER_MCP.value,
            mp_hands.HandLandmark.PINKY_MCP.value
        ]
        return world_pts[idx].mean(axis=0)

    @staticmethod
    def _unit(v, eps=1e-9):
        v = np.asarray(v, dtype=np.float64)
        return v / (np.linalg.norm(v) + eps)

    @classmethod
    def _palm_normal(cls, world_pts, mp_hands):
        """手掌平面法向量:由 wrist->index_mcp 与 wrist->pinky_mcp 叉乘得到。"""
        LM = mp_hands.HandLandmark
        wrist = world_pts[LM.WRIST.value]
        idx_mcp = world_pts[LM.INDEX_FINGER_MCP.value]
        pinky_mcp = world_pts[LM.PINKY_MCP.value]
        return cls._unit(np.cross(idx_mcp - wrist, pinky_mcp - wrist))

    @classmethod
    def _abduction_angle(cls, world_pts, mp_hands, mcp_idx, ref_vec, palm_n):
        """
        外展角(AA):手指近节骨(mcp->pip)投影到手掌平面后,相对参考轴 ref_vec 的带符号夹角(度)。
        ref_vec 一般取"手掌纵轴"(wrist->middle_mcp)。符号用 palm_n 定向,区分向拇指侧/小指侧张开。
        """
        pip_idx = mcp_idx + 1
        bone = world_pts[pip_idx] - world_pts[mcp_idx]
        # 投影到手掌平面(去掉法向分量)
        bone_in_plane = bone - np.dot(bone, palm_n) * palm_n
        ref_in_plane = ref_vec - np.dot(ref_vec, palm_n) * palm_n
        bone_u = cls._unit(bone_in_plane)
        ref_u = cls._unit(ref_in_plane)
        cos = np.clip(np.dot(bone_u, ref_u), -1.0, 1.0)
        ang = float(np.degrees(np.arccos(cos)))
        # 带符号:用叉乘在 palm_n 上的投影定正负
        sign = np.sign(np.dot(np.cross(ref_u, bone_u), palm_n))
        return ang * (sign if sign != 0 else 1.0)

    @classmethod
    def _compute_joint_angles(cls, world_pts, mp_hands):
        """
        计算 20 个关节角(emg2pose 定义):每指 3 个屈曲角(FE) + 1 个外展角(AA)。
        屈曲角沿用原几何夹角算法;外展角用手掌平面投影法(几何近似)。
        返回 dict,键与 self.joint_names 一致(顺序无关,取用时按 joint_names 索引)。
        """
        LM = mp_hands.HandLandmark
        ang = {}
        pc = cls._palm_center(world_pts, mp_hands)
        palm_n = cls._palm_normal(world_pts, mp_hands)
        # 手掌纵轴参考:wrist -> middle_mcp
        ref_axis = world_pts[LM.MIDDLE_FINGER_MCP.value] - world_pts[LM.WRIST.value]

        # ---- 拇指(4 DOF): CMC_FE, CMC_AA, MCP_FE, IP_FE ----
        wrist = LM.WRIST.value
        th_cmc, th_mcp, th_ip, th_tip = (LM.THUMB_CMC.value, LM.THUMB_MCP.value,
                                         LM.THUMB_IP.value, LM.THUMB_TIP.value)
        ang['THUMB_CMC_FE'] = cls._angle_3pts(world_pts[wrist], world_pts[th_cmc], world_pts[th_mcp])
        # 拇指 CMC 外展:cmc->mcp 骨段相对手掌纵轴在掌面内的偏角
        ang['THUMB_CMC_AA'] = cls._abduction_angle(world_pts, mp_hands, th_cmc, ref_axis, palm_n)
        ang['THUMB_MCP_FE'] = cls._angle_3pts(world_pts[th_cmc], world_pts[th_mcp], world_pts[th_ip])
        ang['THUMB_IP_FE']  = cls._angle_3pts(world_pts[th_mcp], world_pts[th_ip], world_pts[th_tip])

        # ---- 其余四指(各 4 DOF): MCP_AA, MCP_FE, PIP_FE, DIP_FE ----
        def finger(name, mcp):
            pip, dip, tip = mcp + 1, mcp + 2, mcp + 3
            ang[f'{name}_MCP_AA'] = cls._abduction_angle(world_pts, mp_hands, mcp, ref_axis, palm_n)
            ang[f'{name}_MCP_FE'] = cls._angle_3pts(pc, world_pts[mcp], world_pts[pip])
            ang[f'{name}_PIP_FE'] = cls._angle_3pts(world_pts[mcp], world_pts[pip], world_pts[dip])
            ang[f'{name}_DIP_FE'] = cls._angle_3pts(world_pts[pip], world_pts[dip], world_pts[tip])

        finger('INDEX',  LM.INDEX_FINGER_MCP.value)
        finger('MIDDLE', LM.MIDDLE_FINGER_MCP.value)
        finger('RING',   LM.RING_FINGER_MCP.value)
        finger('PINKY',  LM.PINKY_MCP.value)
        return ang

    @staticmethod
    def _draw_angle_labels(img, lms2d, angles, mp_hands):
        def put(text, idx, dx=0, dy=-10):
            x, y = lms2d[idx]
            cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)

        LM = mp_hands.HandLandmark
        # 拇指:CMC 屈曲/外展 + MCP + IP
        put(f'CMC {angles["THUMB_CMC_FE"]:.0f}/{angles["THUMB_CMC_AA"]:.0f}', LM.THUMB_CMC.value)
        put(f'MCP {angles["THUMB_MCP_FE"]:.0f}', LM.THUMB_MCP.value)
        put(f'IP  {angles["THUMB_IP_FE"]:.0f}',  LM.THUMB_IP.value)

        for name, base in [('INDEX', LM.INDEX_FINGER_MCP.value),
                           ('MIDDLE', LM.MIDDLE_FINGER_MCP.value),
                           ('RING',   LM.RING_FINGER_MCP.value),
                           ('PINKY',  LM.PINKY_FINGER_MCP.value if hasattr(LM, 'PINKY_FINGER_MCP') else LM.PINKY_MCP.value)]:
            # MCP 显示 屈曲/外展 两个角
            put(f'{name} MCP {angles[f"{name}_MCP_FE"]:.0f}/{angles[f"{name}_MCP_AA"]:.0f}', base)
            put(f'PIP {angles[f"{name}_PIP_FE"]:.0f}', base + 1)
            put(f'DIP {angles[f"{name}_DIP_FE"]:.0f}', base + 2)



if __name__ == "__main__":
    tracker = MediaPipeHandTracker(
        camera_index=0,
        mirror=False,
        show_window=True,   # okay now, because the main loop below calls poll_gui()
        use_2D_coord_for_angles=True
    )
    tracker.start()

    try:
        while True:
            right_hand_angles = tracker.get_mediapipe_angles()
            print(" ".join(
                f"{name}: {angle:.2f}"
                for name, angle in zip(tracker.joint_names, right_hand_angles)
            ))

            if tracker.show_window:
                if not tracker.poll_gui():
                    break

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
