# M6-4 ｜ 多线程推理管线：重叠预处理·推理·后处理

## 结论

| 方案 | FPS | P50 | P95 | 加速比 |
| :-- | :-- | :-- | :-- | :-- |
| 串行 | 86.87 | 9.56 ms | 19.52 ms | 1.00× |
| 双线程 | 95.82 | 52.54 ms | 111.64 ms | 1.10× |
| **三线程** | **157.54** | 34.57 ms | 60.79 ms | **1.81×** |

- **加速比 1.81×**（要求 ≥1.6×），达到理论上限 158.7 FPS 的 **99.3%**
- 三级耗时：pre 3.73 / infer 6.30 / post 0.88 ms，瓶颈在推理
- 14 个单测全绿（2.4 秒），全部用 Mock 引擎，**不依赖真实模型与摄像头**
- 3 分钟稳定性：见 `results/rss_curve.png`

## 目录

```
M6-4/
├── pipeline/
│   ├── buffers.py     # 缓冲池（索引传递，零拷贝）
│   ├── engine.py      # 引擎单例 + 锁
│   ├── stages.py      # 三级阶段（原地写，零分配）
│   └── pipeline.py    # 装配 + 背压 + 优雅退出
├── pipeline.py        # CLI：--threads 1|2|3 --stream camera
├── tools/
│   ├── benchmark_pipeline.py
│   └── watch_rss.py
├── tests/             # 4 个文件 14 个用例
├── results/           # pipeline_benchmark.yaml / rss_curve.png
└── docs/              # design.md / zero_copy.md / bottleneck.md
```

## 跑起来

```bash
# 三线程实时演示（含画框，GPU）
bash ../pycuda.sh pipeline.py --threads 3 --stream camera --device gpu

# 三方案对比（600 帧，结果写 yaml）
bash ../pycuda.sh tools/benchmark_pipeline.py \
    --source ../M2-3/data/raw --frames 600 --device gpu --queue-size 8

# 3 分钟内存曲线
python tools/watch_rss.py --seconds 180 --run-pipeline

# 单测
python -m pytest tests/ -q
```

## 四个必答问题的答案（详见 docs/）

**背压为什么用「阻塞」而不是「丢帧」？**
比赛场景丢一帧就可能漏掉一个目标，而且丢帧会让「帧号↔结果」错位，
后面接追踪（M2-4）时 ID 会乱。阻塞只是延迟升高，数据是完整的。

**`maxsize` 设 1/4/16 分别怎样？**
容量决定「允许多少帧在飞」。1 几乎没有缓冲、延迟最低但最容易让预处理等；
4~8 是吞吐/延迟的平衡区；16 吞吐不再提升（受最慢一级限制），
只是把帧堆在队里、把 P95 拉高。

**消费者比生产者慢时怎么观察到？**
队列长度持续等于 maxsize，且生产端阻塞在 `put`。
本项目在 `test_backpressure.py` 里用采样线程记录 `qsize`，并断言它不超过 maxsize。

**怎么证明「没在拷贝」？**
`buffers.evidence()` 的地址表在跑 100 帧前后完全一致，分配次数恒为 2N；
`np.shares_memory(canvas(i), _canvas[i])` 为 True。见 `docs/zero_copy.md`。

## 三个踩过的坑（现场可能被问）

**1. 双线程模式我先写出了死锁。**
最初把 infer+post 的结果再塞进一个下游有界队列，可那个队列没人消费——
四帧之后推理线程就把自己堵死了。双线程只有两级，不应该有第二个队列。

**2. `q.put()` 在队列满时会无限阻塞，stop 信号传不进来。**
于是生产者永远卡在 put 上、join 永远等不到。改成「先 `put_nowait` 快路径，
满了再进带 0.2 秒超时的循环，每轮检查 stop」。同理 `acquire` 也要带超时。

**3. 无限帧源不能用固定超时 join。**
摄像头是无限流，线程本来就「一直活着」，30 秒超时会把健康状态误判成死锁。
改成正常 `join()`，只在 KeyboardInterrupt 时置停并限时收尾。

## 一个反直觉的发现

`preprocess` 里把「BGR→RGB + /255 + HWC→CHW」写成三次 `np.divide(..., out=)`
之后，预处理从 **32 ms 降到 3.7 ms**（同一张 1279×1706 图）。
原来的 `cvtColor + astype + transpose` 三行会分配约 11 MB/帧，
30 FPS 下就是 330 MB/s 的分配-释放——这部分开销比计算本身还大。
