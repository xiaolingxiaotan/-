# -*- coding: utf-8 -*-
"""正确性：流水线输出 == 串行输出；零拷贝的证据可复现。"""
import numpy as np

from conftest import FakeEngine, fake_post, fake_pre, make_source
from pipeline.buffers import FrameBuffers
from pipeline.pipeline import Pipeline


def _run(mode, n=30, qsize=4):
    buffers = FrameBuffers(n=4, size=64)
    got = []

    def on_result(idx, dets, lat):
        got.append((idx, float(dets[0, 4])))

    pipe = Pipeline(FakeEngine(delay=0.002), buffers, make_source(n),
                    mode=mode, queue_size=qsize,
                    stages={"pre": fake_pre, "post": fake_post},
                    on_result=on_result)
    snap = pipe.run()
    return sorted(got), snap, buffers


def test_dual_matches_serial():
    a, _, _ = _run("serial")
    b, _, _ = _run("dual")
    assert a == b


def test_triple_matches_serial():
    a, _, _ = _run("serial")
    c, _, _ = _run("triple")
    assert a == c


def test_zero_copy_addresses_unchanged():
    """跑 100 帧前后，缓冲池的地址表完全一致 —— 说明运行期没有新建数组。"""
    buffers = FrameBuffers(n=4, size=64)
    before = buffers.evidence()
    pipe = Pipeline(FakeEngine(delay=0.0), buffers, make_source(100),
                    mode="triple", queue_size=4,
                    stages={"pre": fake_pre, "post": fake_post})
    pipe.run()
    after = buffers.evidence()

    assert before["canvas_addrs"] == after["canvas_addrs"]
    assert before["blob_addrs"] == after["blob_addrs"]
    assert after["allocations_total"] == before["allocations_total"] == 8
    assert after["views_share_memory"] is True


def test_buffer_returns_view_not_copy():
    """`canvas(i)` 必须是视图：改它等于改底层内存。"""
    buffers = FrameBuffers(n=2, size=64)
    i = buffers.acquire()
    v = buffers.canvas(i)
    v[0, 0, 0] = 77
    assert buffers.canvas(i)[0, 0, 0] == 77
    assert np.shares_memory(v, buffers.canvas(i))
    buffers.release(i)
