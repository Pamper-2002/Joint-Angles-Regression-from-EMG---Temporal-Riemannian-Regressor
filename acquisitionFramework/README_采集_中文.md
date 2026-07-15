# EMG + 手指关节角度同步采集 — 使用说明(中文)

## 🚀 快速启动(直接复制粘贴)

> 关键:**必须用本项目的虚拟环境 `.venv-emg`**,不要用系统 Python 或别的项目环境。
> 下面的命令用绝对路径调用该环境的 python,不管当前激活的是哪个环境都能跑对。

**PowerShell(即 VSCode 默认终端,提示符是 `PS >`):**
```powershell
cd D:\Project_CJ\rgb2pose\Joint-Angles-Regression-from-EMG---Temporal-Riemannian-Regressor\acquisitionFramework
& "D:\Project_CJ\rgb2pose\.venv-emg\Scripts\python.exe" main_o3d.py
```
> PowerShell 里路径带引号时,前面**必须加 `&`**(调用运算符),否则报 `UnexpectedToken`。

**Git Bash / cmd:**
```bash
cd "D:/Project_CJ/rgb2pose/Joint-Angles-Regression-from-EMG---Temporal-Riemannian-Regressor/acquisitionFramework"
"D:/Project_CJ/rgb2pose/.venv-emg/Scripts/python.exe" main_o3d.py
```

跑起来后会弹出两个窗口:**Open3D 五指手 3D 窗** + **OpenCV RGB 预览窗**。
退出:关 Open3D 窗 / RGB 窗按 ESC 或 q / 控制台 Ctrl+C。

---

本说明记录本机已搭好的环境与运行方法。目标:用 MindRove 8 通道 EMG 手环 + 笔记本摄像头,
同步采集前臂肌电(EMG)与手指关节角度,保存为 `.npy` 供后续训练。

## 一、环境(已搭好,无需重装)

- 独立虚拟环境:`D:\Project_CJ\rgb2pose\.venv-emg`(不影响系统全局 Python)
- 解释器:`D:\Project_CJ\rgb2pose\.venv-emg\Scripts\python.exe`
- 关键依赖版本(已验证可共存):
  - `mediapipe==0.10.14`(含旧版 `mp.solutions.hands`,项目依赖)
  - `tensorflow==2.18.1`(锁此版本;2.21 会拉高 protobuf 与 mediapipe 硬冲突,**切勿升级**)
  - `protobuf 4.25.9`(两者兼容交集)、`numpy<2`
  - `open3d 0.19.0`(五指实体手 3D 渲染,有 py3.12 预编译 wheel)
  - `mindrove 5.3.0`、`pyriemann 0.12`、`scikit-learn`、`scipy`、`mne`、`joblib`
  - `pybullet 3.2.7`(原 LEAP Hand 遗留依赖,当前 main_o3d.py 已不用;保留不影响)
- 完整清单见同目录 `requirements.txt`

VC++ 编译器已装:`Microsoft Visual Studio 2022 BuildTools`(当初为编译 pybullet 装的;现虽不用 pybullet,保留无害)。

## 二、采集前准备

1. **打开 MindRove 手环电源**,戴到前臂。
2. **电脑连接手环的 WiFi 热点**(手环自身是 AP;单设备时 `MindRoveInputParams()` 留空即可,
   本机实测连到 `192.168.4.1:4210`)。
3. 确认摄像头可用(本机摄像头 index=0,640x480)。若用外接摄像头,改
   `main_o3d.py` 里 `MediaPipeHandTracker(camera_index=0, ...)` 的 `camera_index`。
4. 输出目录 `savedData/`、`savedModel/` 已建好(保存时必须存在,否则报错)。

## 三、运行采集

唯一入口是 **`main_o3d.py`**(五指实体手 + RGB 预览)。在**普通终端**
(cmd 或 PowerShell,需有 stdin 才能用 s/t 命令)运行:

```bat
cd D:\Project_CJ\rgb2pose\Joint-Angles-Regression-from-EMG---Temporal-Riemannian-Regressor\acquisitionFramework
D:\Project_CJ\rgb2pose\.venv-emg\Scripts\python.exe main_o3d.py
```

启动后弹出两个窗口:

- **Open3D 窗口**:显示 UmeTrack **LBS 连续蒙皮手**。MediaPipe 21 个世界坐标关键点先变换到
  掌心局部坐标，再由带关节限位和时间先验的 IK 拟合 20 个 UmeTrack 关节弧度，最后经 FK+LBS 驱动网格。
- **OpenCV RGB 预览窗**:摄像头画面 + 手部骨架 + 20 个关节角度数字(每指的 MCP 显示 屈曲/外展 两个值)。

采集与保存(`config.EMG=True` 时):

- 程序启动约 10 秒后,控制台出现 `train (t), save(s) :` 提示:
  - 输入 **`s`** 回车 → 保存当前已采集数据到 `savedData/`
  - 输入 **`t`** 回车 → 用已采数据训练 EMG→关节角回归模型(存 `savedModel/`)
- 退出:关闭 Open3D 窗口 / RGB 窗口按 **ESC** 或 **q** / 控制台 **Ctrl+C**。

> 说明:原版四指 LEAP Hand 入口已弃用。视觉主路径不再把几何角直接套到另一套骨骼轴上；
> 旧的 20 维角度映射仅作为 IK 无法收敛时的兼容回退。

## 四、保存的数据格式(savedData/)

| 文件 | shape | 含义 |
|---|---|---|
| `EMG_data.npy` | (N, 8) | 8 通道 EMG,500Hz,N≈采集秒数×500 |
| `label_data.npy` | (M, 20) | M 帧关节角度,每帧 **20** 个角(见下,对齐 emg2pose) |
| `label_ts_data.npy` | (M,) | 每帧角度对应的 EMG 样本索引(同步时间轴,单调递增) |
| `shown_pred_data.npy` | (M, 20) | 模型预测角度(未加载模型时=当前角度) |
| `train_times.npy` | (K,) | 触发训练时的帧索引 |
| `label_schema.json` | JSON | 标签 schema、单位和 20 个关节名称 |

**20 个关节角顺序**(对齐 emg2pose 的 constants.py;FE=屈曲,AA=外展/内收):
```
0 THUMB_CMC_FE   1 THUMB_CMC_AA   2 THUMB_MCP_FE   3 THUMB_IP_FE
4 INDEX_MCP_AA   5 INDEX_MCP_FE   6 INDEX_PIP_FE   7 INDEX_DIP_FE
8 MIDDLE_MCP_AA  9 MIDDLE_MCP_FE 10 MIDDLE_PIP_FE 11 MIDDLE_DIP_FE
12 RING_MCP_AA  13 RING_MCP_FE   14 RING_PIP_FE   15 RING_DIP_FE
16 PINKY_MCP_AA 17 PINKY_MCP_FE  18 PINKY_PIP_FE  19 PINKY_DIP_FE
```
> 当前 schema 为 `umetrack20_rad_v1`，单位是**弧度**，关节轴与渲染骨架完全一致。
> 没有 schema 文件的旧模型按 `mediapipe_geometry20_deg_v1` 处理，并经过修正后的兼容映射；
> 新旧数据不可直接拼接训练。
>
> **注意**:数据格式已从原 15 维升级为 20 维,旧的 15 维 .npy 与新格式不兼容。

**同步原理**:EMG 是 500Hz 密集流,关节角是稀疏标签。每个新的视觉时间戳只记录一次，
`labelTs = 当前EMG样本总数`,把角度"钉"在 EMG 时间轴的对应位置;训练时线性插值补齐
(见 `EMG_regression.py` 的 `labelInterpolation`)。

## 五、配置开关(config.py)

| 项 | 默认 | 说明 |
|---|---|---|
| `EMG` | True | 是否采集 EMG。设 False 可只跑手部追踪+五指手(无需手环,调试用) |
| `PRELOAD_EMG_MODEL` | False | 是否加载预训练模型做实时预测 |
| `MEDIAPIPE_MAX_NUM_HANDS` | 1 | 只检测一只手，减少无效推理 |
| `MEDIAPIPE_MODEL_COMPLEXITY` | 0 | 实时优先的轻量模型 |
| `HAND_IK_ITERATIONS` | 16 | 每个新姿态的 IK 最大迭代数 |
| `HAND_IK_CONVERGENCE_RMSE` | 0.08 | 归一化关键点 RMSE 收敛阈值 |
| `HAND_RENDER_SUBDIVIDE` | 0 | 0 为实时原拓扑；1 为高质量但更耗 CPU |
| `HAND_RENDER_SMOOTHING` | 0.5 | 按真实时间间隔计算的渲染平滑强度 |
| `VIRTUAL` / `PHYSICAL` | - | **已废弃**:原 LEAP Hand 开关,main_o3d.py 不再读取(固定用 Open3D 五指手) |

其它 `FREQ_BANDS`/`EMG_WINDOW_*`/`EMG_SEQUENCE_LENGTH` 是 EMG 特征窗口参数(按 500Hz 标定),训练时用。
摄像头 index 在 `main_o3d.py` 里 `camera_index=0`;非 0 号摄像头改这里。

## 六、已验证结论

- MindRove 真机连接,`getEMG()` 返回 (8,1000)@2秒,500Hz ✓
- 同步采集 → EMG (N,8) + 角度 (M,**20**) + labelTs (M,) 单调递增 ✓
- MediaPipe 原子发布 21 点、左右手、时间戳；同一帧不会重复求解或重复记录 ✓
- 掌心局部归一化 + 受约束 UmeTrack IK 输出 20 个 `umetrack20_rad_v1` 关节弧度 ✓
- Open3D 原拓扑网格就地更新；默认不再每帧细分和重建网格 ✓
- 控制台分别报告 `capture`、`inference`、`right_pose` FPS；后者才是有效右手姿态率 ✓

> 提示:真机采集前确认 MindRove 已开机、电脑已连设备 WiFi 热点。若报
> `unable to prepare streaming session`,通常是设备休眠或未连上热点。

## 七、注意事项

- 运行时会有一条 protobuf `GetPrototype() is deprecated` 警告,**无害**,可忽略。
- TensorFlow 仅在采集导入与 `t` 训练时用到;采集本身默认 CPU,不需要 GPU。
- `Network problem, socket error 10060` 是 MindRove 网络/热点超时，不是 MediaPipe 或 IK 的 FPS 错误；
  请检查电脑是否连接手环热点、设备是否休眠以及 `192.168.4.1:4210` 是否可达。
- 后处理脚本 `numpyToFif.py` / `mne_visualizer.py` 有既有 bug(引用未生成的
  `label_confidence_data.npy`、`name=...` 占位符),需要时再单独修,与采集无关。

## 八、测试与诊断

在仓库根目录执行：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& "D:\Project_CJ\rgb2pose\.venv-emg\Scripts\python.exe" -m pytest -q
```

判断卡顿时优先看控制台三项：`capture` 低通常是相机/驱动，`inference` 明显低于 `capture`
通常是 MediaPipe CPU 推理，`right_pose` 低于 `inference` 则多为画面没有稳定识别到右手。
IK 和渲染只消费**新时间戳**，主循环轮询频率不会被误记成姿态 FPS。
