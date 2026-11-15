# M6-1 板端环境搭建（从裸板开始）

> 本项目实际使用的运行环境：**笔记本 + RTX 5060 Laptop GPU + WSL2(Ubuntu 22.04)**。
> 题面要求「自备开发板」，本机没有 Jetson / RK3588 / RDK 之类的板子，
> 因此三模式中的 **NPU 记为「设备不具备」**（题面 §2.3 允许这样记录），
> CPU 与 GPU(CUDA) 两种模式均为**实测数据**，不是估算。

---

## 1. 环境清单

| 项 | 版本 | 说明 |
| :-- | :-- | :-- |
| 系统 | Ubuntu 22.04 (WSL2) | 内核 6.18，Windows 11 宿主 |
| Python | 3.10.12 | 虚拟环境 `.venv`（uv 创建） |
| onnxruntime-gpu | 1.23.2 | 含 `CUDAExecutionProvider` / `TensorrtExecutionProvider` |
| CUDA | 12.8（torch 自带的 nvidia/* 运行库） | 驱动由 Windows 侧提供 |
| OpenCV / numpy | 5.0.0 / 2.2.6 | 仅预处理与画框 |

**推理端不装 PyTorch**：`infer.py` 只依赖 `onnxruntime + opencv + numpy`。
`tools/check_consistency.py` 与 `export_onnx.py` 需要 PyTorch，它们属于「训练侧工具」，
不属于部署链路。

验证「零训练框架依赖」：

```bash
# 把 torch 的 import 拦掉，再跑一次推理
python - <<'PY'
import sys
class Block:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] == "torch" else None
    def load_module(self, name):
        raise ImportError("torch 被拦截")
sys.meta_path.insert(0, Block())
sys.argv = ["infer.py", "--model", "models/best_fp32.onnx", "--source", "test.jpg"]
exec(open("infer.py", encoding="utf-8").read())
PY
```

---

## 2. 从零安装

```bash
# 1) 建环境
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python \
    onnxruntime-gpu==1.23.2 numpy opencv-python pyyaml psutil

# 2) 验证执行器
.venv/bin/python -c "import onnxruntime as o; print(o.get_available_providers())"
# 期望：[..., 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 3) 让 onnxruntime 找到 CUDA 运行库
#    坑：onnxruntime 不会自动去 site-packages/nvidia/*/lib 里找 libcudnn/libcublas，
#    必须把这些目录塞进 LD_LIBRARY_PATH，否则 CUDA EP 会「静默退化」成 CPU EP
#    （会话能建、也能跑，但 get_providers()[0] 是 CPU —— 性能数据全是错的）。
for d in .venv/lib/python3.10/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done
```

`pycuda.sh` 就是上面这段的封装，用它启动任何需要 GPU 的脚本：

```bash
bash pycuda.sh tools/benchmark.py --model models/best_fp32.onnx --device gpu --seconds 60
```

---

## 3. 三模式状态

| 模式 | 执行器 | 状态 | 依据 |
| :-- | :-- | :-- | :-- |
| cpu | `CPUExecutionProvider` | ✅ 实测 | `results/benchmark_cpu_1114.yaml` |
| gpu | `CUDAExecutionProvider` | ✅ 实测 | `results/benchmark_gpu_1114.yaml` |
| npu | — | ❌ 设备不具备 | 本机无 NPU；`onnxruntime` 也没有对应 EP。按题面 §2.3「跑不起来怎么记录」处理：显式记为不可用，不伪造数据 |

---

## 4. 两个踩过的坑

**坑 1：CUDA EP 静默退化。**
`providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` 这种写法，在 CUDA 库缺失时
不会报错，而是安静地退回 CPU。表现是「GPU 模式 15 FPS，比 CPU 还快一点点」——
很容易被当成「GPU 就这样」写进报告。**判定方法：打印 `sess.get_providers()[0]`**，
不是 CUDA 就说明没生效。

**坑 2：冷启动 45 秒。**
第一次跑 CUDA EP 时 cuDNN 会做算法自动选择（autotuning），单次耗时 **45.9 秒**，
是稳态 7.47 ms 的 **6154 倍**。如果基准测试把第一帧算进平均，60 秒里只有十几帧，
FPS 会被严重低估。`tools/benchmark.py` 的做法是：先单独跑一帧记为 `latency_cold_ms`，
再开始计时，冷启动不计入统计。
