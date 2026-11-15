# -*- coding: utf-8 -*-
"""背压：队列满时生产者必须**等待**，而不是丢帧或无限增长。"""
import time

import numpy as np

from conftest import FakeEngine, make_source
from pipeline.buffers import FrameBuffers
from pipeline.pipeline import Pipeline


def test_queue_is_bounded_and_no_drop(stages):
    """消费很慢时：队列长度不超过 maxsize，且**一帧都不丢**。"""
    buffers = FrameBuffers(n=4, size=64)
    eng = FakeEngine(delay=0.02)            # 推理比生产慢 10 倍
    pipe = Pipeline(eng, buffers, make_source(40), mode="triple",
                    queue_size=2, stages=stages)

    # 边跑边采样队列长度
    qmax = [0]

    def sampler():
        while not pipe.stop.is_set() and pipe.stats.frames_out < 40:
            qmax[0] = max(qmax[0], pipe.q_pre.qsize(), pipe.q_post.qsize())
            time.sleep(0.002)

    import threading
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    snap = pipe.run()
    pipe.stop.set()

    assert qmax[0] <= 2, "队列超过了 maxsize：%d" % qmax[0]
    assert snap["frames_out"] == 40, "丢帧了：%s" % snap
    assert snap["drops"] == 0, "有丢帧：%d" % snap["drops"]


def test_producer_blocks_when_pool_exhausted(stages):
    """缓冲池被占满时，预处理线程必须阻塞（acquire 等待），不会越界。"""
    buffers = FrameBuffers(n=2, size=64)
    eng = FakeEngine(delay=0.05)
    pipe = Pipeline(eng, buffers, make_source(8), mode="triple",
                    queue_size=4, stages=stages)
    snap = pipe.run()
    # 缓冲没漏：跑完后全部归还
    assert buffers.free_count() == 2, "运行结束还有缓冲没归还"
    assert snap["frames_out"] == 8


def test_slow_stage_does_not_grow_buffer_usage(stages):
    """人为拖慢后处理，缓冲池索引不应泄漏（acquire 次数 == release 次数）。"""
    buffers = FrameBuffers(n=3, size=64)
    eng = FakeEngine(delay=0.001)
    pipe = Pipeline(eng, buffers, make_source(30), mode="triple",
                    queue_size=3, stages=stages)
    snap = pipe.run()
    ev = buffers.evidence()
    assert snap["frames_out"] == 30
    assert buffers.free_count() == 3
    assert ev["allocations_total"] == 6      # 2*n，运行期没有新增分配
