# M6-2 ｜ 量化与硬件加速

## 一句话结论

**在本机（ORT 1.23.2 + YOLOv8n）上，静态 QDQ 量化在 `quantize_static` 阶段硬崩溃、
动态量化产出的模型 CPU 后端不支持，所以「量化后 mAP/FPS 对比」这一项做不出可信数字。**
我做了两件事替代：跑通动态量化并测出它为什么跑不动，以及用「逐层权重 INT8 敏感度」
定位量化损失的来源。下面把失败原因、实验数据和结论都摆出来，不含糊过去。

| 项目 | 状态 | 说明 |
| :-- | :-- | :-- |
| 校准集（100~500 张，独立于训练集） | ✅ | val+test 共 118 张 |
| 动态量化 | ⚠️ 跑通但不可用 | 12.16 → 3.31 MB，但模型含 `ConvInteger`，CPU EP 无实现 |
| 静态量化为 QDQ | ❌ 硬崩溃 | `quantize_static` 无异常直接 abort，见 §3 |
| 静态量化为 QOperator | ❌ 同上 | 两种格式、per_channel 开关、仅量化 Conv 都试过 |
| 逐层精度分析 | ✅ | 64 层卷积的 INT8 敏感度，CSV + 图 |
| 平台加速器（TensorRT） | ❌ 缺库 | `libnvinfer.so.10` 不存在，EP 静默退化到 CPU |
| 逐类报告 | ⚠️ 无量化模型可测 | 只有浮点模型的逐类指标（见 M2-3） |

## 1. 校准集

题面要求「100~500 张真实图像，**独立于训练集**」。用 `data/splits/val.txt` +
`test.txt` 的 118 张，训练图一张没用——用训练图校准会让量化参数"见过答案"，
实测精度虚高（题面 §2.1 第 1 条的原话）。

校准流程与推理严格一致：letterbox 到 640×640、RGB、`/255`、NCHW。
这一段复用了 `tools/quantize.py::CalibReader`，与部署端同一份代码。

## 2. 动态量化：跑通，但模型跑不起来

```bash
python tools/quantize.py          # 默认只做动态量化 + 逐层分析
```

结果：`best_int8_dynamic.onnx` **3.31 MB**（浮点 12.16 MB，压到 27%）。

但把它加载起来直接报错：

```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
node with name '/model.0/conv/Conv_quant'
```

原因：`quantize_dynamic` 只量化**权重**，激活在运行时量化，它生成的卷积是
`ConvInteger`——这个算子只有部分后端（如某些 QNN/NNAPI）实现，
**onnxruntime 的 CPU EP 没有**。

这就是题面说的「动态量化简单，加速有限」在本模型上的真实形态：
不是"慢一点"，而是**根本跑不了**。对 CNN 来说动态量化基本没有实用价值，
它本来是给 Transformer/LSTM 的 MatMul、Gemm 设计的。

## 3. 静态量化：`quantize_static` 硬崩溃

```bash
python tools/quantize.py --static
```

现象：打印完 `Please consider to run pre-processing before quantization`
之后进程**直接终止**——没有 Python 异常、没有 traceback，退出码 1。

试过并全部失败在同一位置的组合：

| 组合 | 结果 |
| :-- | :-- |
| `QuantFormat.QDQ` + `per_channel=True` | 崩溃 |
| `QuantFormat.QOperator` + `per_channel=False` | 崩溃 |
| `op_types_to_quantize=["Conv"]`（只量化卷积） | 崩溃 |
| 先 `quant_pre_process` 再量化 | `quant_pre_process` 先报 `Incomplete symbolic shape inference` |
| 校准图 4 / 40 / 118 张 | 与图数无关，同样崩 |

**归因**：YOLOv8 的导出图里有 `Resize`（带动态 scale）、`Slice`/`Concat` 等
shape 相关算子，ONNX Runtime 在为它们做符号推导 + 量化节点插入时崩在 C++ 层。
这是 ORT 1.23.2 对这类图的已知薄弱点，不是参数没调对。

要绕过它有三条路，都需要额外环境，今晚做不完，写在这里作为改进方向：
1. 用 ONNX Runtime 1.17/1.18（对 YOLO 图的量化支持更稳）；
2. 改用 `onnxsim` 先把图简化成静态 shape 再量化；
3. 用 TensorRT 的 `trtexec --int8 --calib` 做平台量化（需要先装 TensorRT）。

## 4. 逐层精度分析（这一项做成了）

既然拿不到可运行的量化模型，就换一个更能定位问题的实验：
**把某一层的权重单独压成 INT8 再还原，其余层保持浮点，看输出掉多少。**
这样误差只归因到这一层，不会被其他层累积掩盖。

```bash
python tools/layer_sensitivity.py --n-img 4
# -> results/layer_sensitivity.csv / .png
```

结果（64 层卷积，逐层单独量化）：

| 层 | 参数量 | CosSim | max\|Δ\| |
| :-- | :-- | :-- | :-- |
| `model.0.conv`（第一层） | 432 | 0.999978 | 34.15 |
| `model.1.conv` | 4608 | 0.999985 | 26.79 |
| `model.2.m.0.cv1.conv` | 2304 | 0.999993 | 27.19 |
| …（中间层） | — | 0.99999x | 10~20 |
| `model.22.cv3.*`（检测头分类支路） | — | 1.000000 | 很小 |

**两个结论：**

1. **没有哪一层"背锅"**。单独量化任何一层，输出 CosSim 都在 0.99997 以上。
   说明权重 per-channel INT8 对 YOLOv8 几乎无损——这也解释了为什么业界
   INT8 量化的精度损失普遍能做到 <1%：损失不在权重，而在**激活量化**。

2. **越靠前越敏感**，且规律很清楚：第一层是 3 通道输入的直接卷积，
   量化误差没有机会被后续层"平均掉"；越往后通道越多、特征越冗余，
   单层扰动对输出的影响被稀释。检测头最后的分类支路最不敏感（1.000000），
   因为它们只做 1×1 卷积、输入已经是高度抽象的特征。

3. `max|Δ|` 看起来很大（34.15）但 CosSim 几乎为 1，这不矛盾：
   输出张量里有个别元素（置信度 logits 的极值）变化大，但整体分布没变。
   **判断量化影响要看 CosSim / 分布，不能只看 max|Δ|**——这一点在
   M6-1 的一致性验证里也是一样的道理。

## 5. 改进方向（按性价比）

1. **换 ORT 版本或先 onnxsim**：最可能让静态量化跑通，成本约 1 小时。
2. **装 TensorRT 做平台量化**：`libnvinfer` 装上后 `trtexec --int8` 走校准，
   精度和速度通常都比 ORT 的通用量化好；本机有 RTX 5060，条件是具备的。
3. **激活量化用百分比（Percentile）而非 MinMax**：YOLO 的输出动态范围大，
   MinMax 容易被离群值拉偏——等量化能跑通后再对比这两种校准方法。

## 目录

```
M6-2/
├── tools/
│   ├── quantize.py            # 动态量化 + 校准集（--static 可选，本机会崩）
│   └── layer_sensitivity.py   # 逐层 INT8 敏感度（本机可跑的那条路）
├── models/
│   └── best_int8_dynamic.onnx # 3.31 MB（CPU EP 不支持 ConvInteger）
└── results/
    ├── layer_sensitivity.csv  # 逐层敏感度（64 层卷积）
    └── layer_sensitivity.png  # Top 25 条形图
```

> 注：`quant_analysis.csv`（浮点 vs 量化模型的逐层 CosSim）没有产出——
> 它依赖一个**能跑起来的量化模型**，而本机产出的两个都跑不起来（§2、§3）。
> 逐层分析改用 `layer_sensitivity.csv`，结论更强：直接归因到单层。
