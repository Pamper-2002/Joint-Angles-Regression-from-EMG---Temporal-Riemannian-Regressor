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

- **Open3D 窗口**:显示完整**五指实体手**(21 个关节球 + 21 根指骨圆柱,肉色蒙皮质感),
  由 MediaPipe 真实 21 个 3D 关键点驱动,可鼠标旋转/缩放视角。首次检测到手时自动对准视角。
- **OpenCV RGB 预览窗**:摄像头画面 + 手部骨架 + 20 个关节角度数字(每指的 MCP 显示 屈曲/外展 两个值)。

采集与保存(`config.EMG=True` 时):

- 程序启动约 10 秒后,控制台出现 `train (t), save(s) :` 提示:
  - 输入 **`s`** 回车 → 保存当前已采集数据到 `savedData/`
  - 输入 **`t`** 回车 → 用已采数据训练 EMG→关节角回归模型(存 `savedModel/`)
- 退出:关闭 Open3D 窗口 / RGB 窗口按 **ESC** 或 **q** / 控制台 **Ctrl+C**。

> 说明:原版四指 LEAP Hand 入口(main.py + pybullet)已弃用并删除。现改用 MediaPipe 21 点
> 直接渲染五指实体手,精度更高(直接用真实关键点,无 LEAP 映射的粗糙近似),且新增小指与外展角。

## 四、保存的数据格式(savedData/)

| 文件 | shape | 含义 |
|---|---|---|
| `EMG_data.npy` | (N, 8) | 8 通道 EMG,500Hz,N≈采集秒数×500 |
| `label_data.npy` | (M, 20) | M 帧关节角度,每帧 **20** 个角(见下,对齐 emg2pose) |
| `label_ts_data.npy` | (M,) | 每帧角度对应的 EMG 样本索引(同步时间轴,单调递增) |
| `shown_pred_data.npy` | (M, 20) | 模型预测角度(未加载模型时=当前角度) |
| `train_times.npy` | (K,) | 触发训练时的帧索引 |

**20 个关节角顺序**(对齐 emg2pose 的 constants.py;FE=屈曲,AA=外展/内收):
```
0 THUMB_CMC_FE   1 THUMB_CMC_AA   2 THUMB_MCP_FE   3 THUMB_IP_FE
4 INDEX_MCP_AA   5 INDEX_MCP_FE   6 INDEX_PIP_FE   7 INDEX_DIP_FE
8 MIDDLE_MCP_AA  9 MIDDLE_MCP_FE 10 MIDDLE_PIP_FE 11 MIDDLE_DIP_FE
12 RING_MCP_AA  13 RING_MCP_FE   14 RING_PIP_FE   15 RING_DIP_FE
16 PINKY_MCP_AA 17 PINKY_MCP_FE  18 PINKY_PIP_FE  19 PINKY_DIP_FE
```
> 相比原版每指多 1 个**外展角(AA)**(手指左右张开),共 5 个。外展角用手掌平面投影几何法计算,
> 是几何近似(非 emg2pose 原生旋转轴定义),用于本框架自采自训一致即可,不建议与 emg2pose 官方数据集混用。
>
> **注意**:数据格式已从原 15 维升级为 20 维,旧的 15 维 .npy 与新格式不兼容。

**同步原理**:EMG 是 500Hz 密集流,关节角是~30Hz 稀疏标签。每次采集步记录
`labelTs = 当前EMG样本总数`,把角度"钉"在 EMG 时间轴的对应位置;训练时线性插值补齐
(见 `EMG_regression.py` 的 `labelInterpolation`)。

## 五、配置开关(config.py)

| 项 | 默认 | 说明 |
|---|---|---|
| `EMG` | True | 是否采集 EMG。设 False 可只跑手部追踪+五指手(无需手环,调试用) |
| `PRELOAD_EMG_MODEL` | False | 是否加载预训练模型做实时预测 |
| `VIRTUAL` / `PHYSICAL` | - | **已废弃**:原 LEAP Hand 开关,main_o3d.py 不再读取(固定用 Open3D 五指手) |

其它 `FREQ_BANDS`/`EMG_WINDOW_*`/`EMG_SEQUENCE_LENGTH` 是 EMG 特征窗口参数(按 500Hz 标定),训练时用。
摄像头 index 在 `main_o3d.py` 里 `camera_index=0`;非 0 号摄像头改这里。

## 六、已验证结论

- MindRove 真机连接,`getEMG()` 返回 (8,1000)@2秒,500Hz ✓
- 同步采集 → EMG (N,8) + 角度 (M,**20**) + labelTs (M,) 单调递增 ✓
- MediaPipe 输出 20 关节角(顺序对齐 emg2pose,含 5 个外展角)✓
- Open3D 五指实体手(21球+21圆柱)实时渲染 + RGB 预览窗共存 ✓
- 换可视化不影响 EMG 采集与标签(采集逻辑未动)✓

> 提示:真机采集前确认 MindRove 已开机、电脑已连设备 WiFi 热点。若报
> `unable to prepare streaming session`,通常是设备休眠或未连上热点。

## 七、注意事项

- 运行时会有一条 protobuf `GetPrototype() is deprecated` 警告,**无害**,可忽略。
- TensorFlow 仅在采集导入与 `t` 训练时用到;采集本身默认 CPU,不需要 GPU。
- 后处理脚本 `numpyToFif.py` / `mne_visualizer.py` 有既有 bug(引用未生成的
  `label_confidence_data.npy`、`name=...` 占位符),需要时再单独修,与采集无关。
