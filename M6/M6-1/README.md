# M6-1 ｜ 从模型到板端：ONNX 导出 + 基础部署

## 结论先放这

| 项目 | 数值 |
| :-- | :-- |
| 模型 | YOLOv8n（COCO 预训练，微调为 6 类） |
| ONNX 大小 | 12.02 MB（fp32，opset 17，动态 shape） |
| **CosSim（ONNX vs PyTorch）** | **1.00000000** |
| 最大绝对差 | 2.21e-03（浮点结合顺序，见 preprocessing.md） |
| 检测框最大偏差 | **0.0002 px** |
| CPU 端到端 | **13.58 FPS**（P50 74.07 ms / P95 107.35 ms） |
| **GPU（CUDA EP）端到端** | **114.57 FPS**（P50 7.47 ms / P95 17.59 ms） |
| NPU | 设备不具备（本机无 NPU） |

单帧推理（不含预处理）在 CPU 上 P50 = 38.98 ms、GPU 上 7.47 ms，
端到端里预处理占了 32 ms 左右——因为测试图是 1279×1706 的竖幅原图，
每次都要做 letterbox。这是真实的端到端成本，没有拿「只算推理」的数字充数。

## 目录

```
M6-1/
├── export_onnx.py            # 可复现导出（opset 17 + dynamic_axes + model_info.yaml）
├── models/
│   ├── best_fp32.pt          # 微调后的权重（对比用）
│   ├── best_fp32.onnx        # 交付模型
│   ├── best_fp32_static.onnx # 静态 shape 对照（回答 §2.1 追问）
│   └── model_info.yaml       # 输入/输出/算子清单
├── infer.py                  # 统一推理（零训练框架依赖）
├── tools/
│   ├── check_consistency.py  # CosSim / 最大绝对差 / 框偏差
│   └── benchmark.py          # 60 秒基准，输出 yaml
├── results/
│   ├── benchmark_cpu_1114.yaml
│   ├── benchmark_gpu_1114.yaml
│   └── consistency.json
└── docs/
    ├── deploy.md             # 板端环境（含两个踩坑）
    └── preprocessing.md      # 预处理逐项对照 + 一致性数据
```

## 一键复现

```bash
# 1. 导出（需要 PyTorch，训练侧工具）
python export_onnx.py --weights models/best_fp32.pt --out models/best_fp32.onnx

# 2. 一致性（需要 PyTorch）
python tools/check_consistency.py --onnx models/best_fp32.onnx \
    --pt models/best_fp32.pt --data ../M2-3/data/raw --num 8

# 3. 基准（不需要 PyTorch；GPU 要带 CUDA 库路径）
bash pycuda.sh tools/benchmark.py --model models/best_fp32.onnx \
    --device cpu --seconds 60
bash pycuda.sh tools/benchmark.py --model models/best_fp32.onnx \
    --device gpu --seconds 60

# 4. 推理（不需要 PyTorch）
python infer.py --model models/best_fp32.onnx --source img.jpg --out dets.json
python infer.py --model models/best_fp32.onnx --device gpu --source camera
```

## 三个必答问题的答案

**opset=17 有什么讲究？换 11 会怎样？**
opset 是算子集的「方言版本」。17 覆盖了 YOLOv8 导出所需的全部算子
（`NonMaxSuppression` 之外的 `Resize/Split/Sigmoid/Conv` 都在 11 里就有，
真正卡住的是 17 之后对 `Slice/Pad` 的动态轴支持更完整）。降到 11 的直接后果是
动态 shape 的 `Reshape/Slice` 会导出失败或退化成固定尺寸分支，需要手工改图。
升到 20+ 则要求 onnxruntime ≥ 1.17，老板子的运行时可能不认。

**dynamic_axes 打开后性能受什么影响？**
见 preprocessing.md 的实测：CPU EP 上动态与静态差 1.3 ms（噪声内）；
真正的代价在 GPU —— 动态 shape 让 **CUDA Graph 无法捕获**（M6-3 要用），
也让 TensorRT 失去固定尺寸的 kernel 选择。所以部署时固定 640 运行，
只在 batch 维保留动态。

**一致性不达标时最可能是哪一步错了？**
按排查顺序：① letterbox 用了 `cv2.resize` 直接拉伸（框会系统偏移）；
② 归一化不一致（少除 255、或多减了 ImageNet 均值）；
③ 通道顺序 BGR/RGB 反了（颜色相关的类别会整体错乱，但框不一定偏）。

## 已知不足

- 没有真实开发板，GPU 模式用的是笔记本的 RTX 5060；NPU 无法测。
- 端到端里预处理占 32 ms（占 CPU 总耗时的 43%），是下一步优化重点
  （M6-4 的多线程流水线就是冲这个来的）。
