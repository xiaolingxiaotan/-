# -*- coding: utf-8 -*-
"""单测公共夹具：**不依赖真实模型**（题面 §2.5 明确要求）。

做法：
  * FakeEngine 实现与 Engine 相同的 `run(blob)` 接口，返回可预测的数组；
  * 假的 pre/post 阶段只做内存操作，耗时可控；
  * 帧源是内存里生成的 numpy 数组，不需要摄像头或视频文件。
"""
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline.buffers import FrameBuffers  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402

SIZE = 64          # 单测用小尺寸，跑得快


class FakeEngine:
    """假引擎：sleep 一段固定时间，返回一个可预测的张量。"""

    def __init__(self, delay=0.005, fail_every=0):
        self.delay = delay
        self.n = 0
        self.fail_every = fail_every
        self.lock = __import__("threading").Lock()

    def run(self, blob):
        time.sleep(self.delay)
        with self.lock:
            self.n += 1
            k = self.n
        if self.fail_every and k % self.fail_every == 0:
            return None
        # 输出里带一个可识别的标量，便于验证「哪一帧对应哪份结果」
        out = np.zeros((1, 10, 4), dtype=np.float32)
        out[0, 0, :] = blob.reshape(-1)[0]      # 把输入的某个值带出来
        return out


def fake_pre(frame, canvas, blob):
    """假预处理：把帧内容写进画布（取重叠区域），张量填一个由帧决定的常数。"""
    h = min(canvas.shape[0], frame.shape[0])
    w = min(canvas.shape[1], frame.shape[1])
    canvas[...] = 0
    canvas[:h, :w] = frame[:h, :w]
    blob[...] = float(frame[0, 0, 0])
    return blob, {"ratio": 1.0, "pad": (0, 0), "orig_hw": frame.shape[:2]}


def fake_post(raw, meta):
    """假后处理：把 raw 里的标量变成「一个框」，便于比对。"""
    v = float(raw[0, 0, 0])
    return np.array([[0.0, 0.0, 10.0, 10.0, v, 0.0]], dtype=np.float32)


def make_source(n_frames, h=48, w=64, seed=0, delay=0.0):
    """内存帧源；每帧的第一个像素值唯一，用来追踪顺序。"""
    rng = np.random.default_rng(seed)
    for i in range(n_frames):
        if delay:
            time.sleep(delay)
        f = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        f[0, 0, 0] = i % 256
        yield i, f


@pytest.fixture
def buffers():
    return FrameBuffers(n=4, size=SIZE)


@pytest.fixture
def stages():
    return {"pre": fake_pre, "post": fake_post}


@pytest.fixture
def make_pipeline(buffers, stages):
    def _make(mode="triple", engine=None, frames=20, **kw):
        return Pipeline(engine or FakeEngine(), buffers,
                        make_source(frames), mode=mode, stages=stages, **kw)
    return _make
