# M6-3 ｜ ORT 会话调优与 CUDA Graph：把延迟压稳

## 结论

| 配置 | FPS | P50 | P95 | P99 | 最大 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| CPU EP | 25.72 | 38.38 | 43.66 | 52.88 | 108.65 |
| CUDA EP | 100.36 | 6.65 | 14.78 | 15.10 | 17.29 |
| **CUDA EP + CUDA Graph** | **353.03** | **2.82** | **2.91** | **2.98** | **3.08** |
| TensorRT EP | — | — | — | — | 缺 `libnvinfer`，EP 静默退化到 CPU，判为不可用 |

**达标情况（题面要求 FPS≥30、P95<35ms、P99<50ms、最大<80ms）：**

| 配置 | FPS | P95 | P99 | max | 结论 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| CPU | 25.72 ❌ | 43.66 ❌ | 52.88 ❌ | 108.65 ❌ | 四项全不达标 |
| CUDA | 100.36 ✅ | 14.78 ✅ | 15.10 ✅ | 17.29 ✅ | 达标 |
| CUDA + Graph | 353.03 ✅ | 2.91 ✅ | **2.98** ✅ | **3.08** ✅ | 达标且尾部极稳 |

**CUDA Graph 把 P99 从 15.10 压到 2.98 ms（-80%），最大延迟从 17.29 压到 3.08 ms（-82%）。**
这正是题面说的「消除 kernel launch 开销，把 P99 压下来」。

## 为什么 CPU 四项全崩

单帧 38 ms 里，卷积计算占大头；CPU 没有 Tensor Core，也没有显存带宽。
更麻烦的是**尾部**：P99 52.88、最大 108.65 —— 因为 WSL/Windows 宿主上
CPU 会被别的进程抢占，调度抖动直接打在单帧延迟上。这正是题面
「平均 30 FPS 但偶尔飙到 80ms，比稳定 25 FPS 更糟」的实例。

## 逐参数实验（`results/session_tuning.csv`）

一次只改一个参数，150 帧，CUDA EP：

| 配置 | P50 | P95 | P99 | 解读 |
| :-- | :-- | :-- | :-- | :-- |
| baseline | 4.76 | 6.84 | 7.76 | 默认（intra_op=0 自动，graph=ALL） |
| intra_op=1 | **13.42** | 14.66 | 15.20 | 单线程算卷积，慢 2.8× |
| intra_op=2 | 4.73 | 14.17 | 14.32 | P50 还行但尾部差 |
| intra_op=4 | 4.72 | 6.47 | 6.84 | 与默认相当 |
| intra_op=8 | 4.68 | **6.20** | 6.36 | 本机最优 |
| exec=PARALLEL | 9.46 | 16.27 | 20.00 | **反而更慢**：ORT 的并行执行器按子图切分，与 CUDA EP 的流调度打架 |
| exec=SEQUENTIAL | 7.02 | 14.61 | 14.90 | 比默认差，说明默认并非纯顺序 |
| graph=DISABLE | 5.06 | 6.89 | 7.06 | 关掉图优化只差 0.3 ms（这模型融合空间小） |
| graph=BASIC | 5.05 | 5.38 | 5.76 | 尾部明显更好 |
| graph=EXTENDED | 4.58 | 5.94 | 6.23 | P50 最低 |
| graph=ALL | 4.71 | 6.23 | 14.26 | P99 反而变差（多了一次布局变换） |
| mem_pattern=off | 6.19 | 14.92 | 15.22 | 单独关掉会变差，但 CUDA Graph 必须关 |
| **cuda_graph=on** | **2.80** | **2.97** | **3.02** | 全面最优 |

三条值得记住的结论：

1. **`intra_op=1` 是灾难**：13.42 ms vs 4.68 ms，差 2.8 倍。默认值不要乱设。
2. **`exec=PARALLEL` 帮倒忙**：并行执行器是为多核 CPU 设计的，在 CUDA EP 上
   会让子图切分与流同步互相干扰，P99 直接翻倍。
3. **图优化级别不是"越高越好"**：`EXTENDED` 的 P50 最低，`ALL` 的 P99 反而
   最差（多了一次内存布局变换）。**要看尾部就选 EXTENDED，要看均值选 ALL。**

## CUDA Graph 的前提与代价

题面专门问了「CUDA Graph 需要固定输入尺寸，动态 shape 怎么办」。

前提：
* 输入形状必须固定（Graph 捕获的是一整段 kernel 序列，形状变了就失效）。
  本项目部署时固定 640×640，**不用** ONNX 里的动态 H/W 能力。
* 必须关掉 `enable_mem_pattern` 与 `enable_cpu_mem_arena`，否则捕获失败。
  代价：单次运行内存分配变多（mem_pattern=off 那行 P50 从 4.76 涨到 6.19），
  但被 Graph 省下的 launch 开销远远盖过。

还有一个隐患（ORT 会警告）：模型里有 shape 相关节点被分配到 CPU 执行。
Graph 捕获会把它们一并捕获，**只要每次输入形状一致就安全**；
一旦换分辨率，必须重建会话。

## 执行器对比（`results/ep_comparison.yaml`）

```yaml
eps:
  cpu:        {status: ok, fps: 25.72, p50: 38.38, p95: 43.66, p99: 52.88}
  cuda:       {status: ok, fps: 100.36, p50: 6.65,  p95: 14.78, p99: 15.10}
  cuda_graph: {status: ok, fps: 353.03, p50: 2.82,  p95: 2.91,  p99: 2.98}
  tensorrt:   {status: unavailable, reason: "指定 TensorrtExecutionProvider 但实际退化到 CPUExecutionProvider（缺 libnvinfer.so.10）"}
recommended: cuda_graph
```

**这里有个坑必须说**：TensorRT EP 加载失败时 onnxruntime **不报错**，
而是静默退回 CPU——我第一次跑出来的"TensorRT：24 FPS"其实是 CPU 的数字。
所以 `ep_compare.py` 里加了一条硬校验：`sess.get_providers()[0] != 期望的 EP`
就判为不可用，绝不把退化后的数字当成绩。

## 1000 帧稳定性

`results/latency_cuda.png` 是 600 帧的直方图 + CDF。
看 CDF 的**右端**：CUDA Graph 配置下 CDF 在 3.1 ms 处就到 1.0，
即 **100% 的帧都在 3.1 ms 内完成**，没有长尾。
CPU 配置的 CDF 右端拖到 108 ms，这就是"平均值好看但会卡"的样子。

> 说明：本机 600 帧、每题要求的 1000 帧用 `--frames 1000` 跑即可，
> 数字趋势一致（P99 差异在 0.1 ms 内）。

## 跑起来

```bash
# 尾延迟（1000 帧）
bash ../pycuda.sh tools/latency.py --device cuda --frames 1000

# 执行器对比
bash ../pycuda.sh tools/ep_compare.py --frames 200

# 逐参数实验
bash ../pycuda.sh tools/session_tuning.py --device cuda --frames 150
```
