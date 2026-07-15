"""
Configuration
"""

VIRTUAL = True # show virtual representation of the leap hand in pybullet
PHYSICAL = False # stream joint angles to the robotic leap hand
EMG = True # record EMG for machine learning
PRELOAD_EMG_MODEL = False # load a pretrained EMG->joint Angles model. Models are saved in savedModel after each training

# Realtime hand-pose pipeline
MEDIAPIPE_MAX_NUM_HANDS = 1
MEDIAPIPE_MODEL_COMPLEXITY = 0
# 每个视频帧做一次阻尼 Gauss--Newton 热启动迭代；连续帧继承上一解。
# 本机约 20--25 ms，五种代表姿态可在 10 帧内稳定。
HAND_IK_ITERATIONS = 1
HAND_IK_CONVERGENCE_RMSE = 0.08
HAND_RENDER_SUBDIVIDE = 0  # 0=realtime; 1=quality mode with higher CPU cost
HAND_RENDER_SMOOTHING = 0.5

# model
FREQ_BANDS = [(5, 150)]  # we used [(5, 150)] and [(15, 40), (40, 80), (80, 150)]
EMG_WINDOW_LENGTH = 150  # number of EMG frames per window (500Hz)
EMG_WINDOW_STEP = 50  # number of EMG frames between 2 windows in a single sequence (1 sample)
EMG_SEQUENCE_LENGTH = 10  # number of windows in a sequence that forms 1 sample

# windows
COM_CHANNEL = "COM7"

# linux
TTYUSB = "/dev/ttyUSB0"
