# M6-4 零拷贝：证据与验证方法

> 题面要求「指出哪一行可能拷贝，并给证据」。本文按「哪里可能拷、怎么证明没拷、拷了会怎样」三段写。

## 一、一帧数据在管线里走的路

| 环节 | 有没有拷贝 | 证据/说明 |
| :-- | :-- | :-- |
| `cap.read()` 产出帧 | 有（**必要**） | 视频解码器必须把像素解出来，这一步无法避免。但它每帧都是新分配，所以不存在「上游复用同一块内存」的隐患 |
| 帧引用传给预处理线程 | 无 | 队列里放的是 Python 对象引用（`(idx, frame)`），不涉及像素拷贝 |
| letterbox 写入画布 | **无（原地写）** | `letterbox_into(img, out, ...)` 用 `out[...] = color` 和 `out[top:top+nh, left:left+nw] = small` 直接写进池子里的 `canvas`；`out` 是预分配的 |
| 画布 → CHW 张量 | **无（原地写）** | `np.divide(canvas[:,:,2], 255.0, out=blob[0,0])` 三次，直接写进池子里的 `blob`。**这是本项目最关键的一行优化** |
| 张量交给 session.run | 无 | 传的是 numpy 数组，ORT 内部按需拷贝到 GPU（这一步在 out-of-process 之外，属于运行时行为） |
| 输出 raw 传后处理 | 无 | 同样是引用 |
| 结果发布 | 无 | 只传 `(idx, dets)`，dets 是几十个 float |
| **缓冲归还** | 无 | 归还的是**索引**（int），不是数据 |

**唯一「本来会拷」的地方是「画布 → CHW 张量」这一步。**
最朴素的写法是：

```python
rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)   # ① 新分配 1.2 MB
blob = rgb.astype(np.float32) / 255.0            # ② 新分配 4.9 MB
blob = np.transpose(blob, (2,0,1))[None]         # ③ 再新分配 4.9 MB（转置不连续，必然复制）
```

三行、三次分配、约 11 MB/帧。30 FPS 下就是 **330 MB/s 的分配与释放**，
GC 压力和内存带宽都被吃掉。本项目的写法把它压成零分配：

```python
np.divide(canvas[:, :, 2], 255.0, out=blob[0, 0])   # 蓝 → 第 0 通道
np.divide(canvas[:, :, 1], 255.0, out=blob[0, 1])   # 绿 → 第 1 通道
np.divide(canvas[:, :, 0], 255.0, out=blob[0, 2])   # 红 → 第 2 通道
```

一次遍历完成「通道互换 + 归一化 + HWC→CHW」，全部写进预分配内存。
实测预处理从 **32 ms 降到 3.7 ms**（同一张 1279×1706 原图），
因为省掉了三次大数组的分配、清零和转置复制。

## 二、怎么证明「没在拷贝」

### 证据 1：地址表不变

```python
before = buffers.evidence()
# ... 跑 100 帧 ...
after = buffers.evidence()
assert before["canvas_addrs"] == after["canvas_addrs"]   # 每块内存首地址完全一致
assert before["blob_addrs"]   == after["blob_addrs"]
```

单测 `tests/test_correctness.py::test_zero_copy_addresses_unchanged` 验证的就是这条。
如果运行期发生了 `np.empty` 之类的分配，拿到的地址一定会变（哪怕偶尔复用，
也不可能 100 帧全程不变）。

### 证据 2：分配次数恒为 2N

`FrameBuffers.n_alloc` 只在构造函数里加 N 次（canvas）+ N 次（blob）= 2N。
跑完 100 帧后这个值**仍然是 2N**，说明运行期没有新增任何帧级分配。

### 证据 3：视图确实共享内存

```python
assert np.shares_memory(buffers.canvas(i), buffers._canvas[i]) is True
```

`canvas(i)` 返回的是切片视图，不是拷贝。

## 三、双缓冲 vs 缓冲池：为什么选后者

题面给了四个方案，本项目选**缓冲池**，理由：

| 方案 | 问题 |
| :-- | :-- |
| 双缓冲 | 两级之间只能有 **1 帧**在飞。三级流水线有**两段**管道，双缓冲需要 4 块内存并手工维护「A/B 交换」的状态机，容易出现「两块都在用」的竞态 |
| 缓冲池 | N 块，谁要谁取，用完归还。天然支持任意级数，且**归还即复用**，不需要状态机 |
| 共享内存 | 面向多进程；本实现是单进程多线程，用不着 |
| `torch.from_numpy` | 会引入 torch 依赖，与「部署端不装训练框架」冲突 |

**如果 Inference 比 Preprocess 慢很多会怎样**：池子会被推理端占满（N 块全在
infer 手里），预处理端 `acquire` 阻塞，整个流水线退化为串行速度——
这是**正确行为**（背压），不是 bug。池子的容量 N 决定了「最多允许多少帧在飞」，
N 越大越不容易卡住，但也越占内存（每组约 5 MB）。

## 四、如果误用了同一块 buffer 会怎样

现象是**偶发的花屏和错框**，而且只在负载高时出现：

1. 预处理线程写第 k 帧到 buffer 0，交给推理；
2. 推理还没算完，预处理又拿了 buffer 0 写第 k+1 帧；
3. 推理算到一半读到一半新一半旧的张量 → 输出一个「不存在的目标」。

因为帧内容接近，错框往往还在合理位置，**看起来像模型抖动**，极难定位。
本实现用「索引 + acquire/release」从根上排除了这种可能：
一块内存只有被 acquire 之后才能写，归还之前别人拿不到。

单测 `tests/test_thread_safety.py::test_buffers_not_written_concurrently`
用「在途计数」验证了同一块 buffer 不会被两个线程同时持有。
