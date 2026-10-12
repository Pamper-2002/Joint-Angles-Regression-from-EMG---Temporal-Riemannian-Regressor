# Data acquisition framework for synchronized EMG and finger joint angles during free gestures

This framework enables to record EMG signals using the [__MindRove 8-channel EMG armband__](https://mindrove.com/product/emg-armband/). If you wish to use a different EMG system, you must adapt the `EMG.py` class

The recording of hand gestures uses [__MediaPipe__](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

The visualization of hand gesture is based on the [__Leap Hand__](https://v1.leaphand.com/)

The framework can be configured using the `config.py` class

The software can then be started using the command
- `python main.py`

After a few seconds, the angles start to be streamed to the chosen visualization.
If EMG are activated in the configuration, EMG are continuously recorded and synchronized with joint angles.
You can save the recorded data by entering `s` in the console, or train a new EMG to joint angles model by entering `t` in the console.
To close the software, press `ESC` in the pybullet window, or press `ctrl+C` in the console.

Finally, `numpyToFif.py` can be used to convert the recorded `.npy` files to `.fif` files, compatible with the package `mne`, and 
`mne_visualizer.py` can be used to visualize the recorded EMG and kinematics signals.

## 中央共享采集（2026-10-12）

首次启动前，在本项目使用的 Python 环境中安装共享包，再安装本项目依赖：

```powershell
python -m pip install -e D:/Project_CJ/device-sdk/sEMG_sdk/shared
```

MindRove Wi-Fi 默认使用中央服务 `semg_shared.mindrove.service`；每个进程只建立 Board -2 订阅，元数据仍查询 master board 0。中央服务自动启动或复用现有 owner，退出本应用只释放自己的订阅。物理 UDP 4210 仅由服务持有；本机 `127.0.0.1:48119` 报告状态，`239.255.42.99:48120` 分发完整 SDK 数据。无数据时检查共享服务与腕带 Wi-Fi，不回退直连。SDK 5.3.0 的 µV/g/deg/s、时间戳及原通道顺序保持不变。

```powershell
python -m semg_shared.mindrove.service --status
```
