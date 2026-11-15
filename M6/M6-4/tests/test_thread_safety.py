# -*- coding: utf-8 -*-
"""线程安全：并发写统计计数不丢、缓冲区不被两个线程同时写。"""
import threading

import numpy as np

from conftest import FakeEngine, fake_post, fake_pre, make_source
from pipeline.buffers import FrameBuffers
from pipeline.pipeline import Pipeline, Stats


def test_stats_counter_no_lost_update():
    """1000 次并发加 1，结果必须正好 1000（GIL 不保证这一点）。"""
    s = Stats()
    n_threads, per_thread = 10, 100

    def worker():
        for _ in range(per_thread):
            s.add_in()

    ts = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert s.frames_in == n_threads * per_thread


def test_buffers_not_written_concurrently():
    """同一块 buffer 不能被两个线程同时持有。

    做法：记录每块 buffer 的「在途标记」，若出现重入就报错。
    """
    buffers = FrameBuffers(n=3, size=64)
    in_use = [0] * 3
    errors = []
    lock = threading.Lock()

    def acquire_tracked():
        i = buffers.acquire()
        with lock:
            in_use[i] += 1
            if in_use[i] > 1:
                errors.append("buffer %d 被重复占用" % i)
        return i

    def release_tracked(i):
        with lock:
            in_use[i] -= 1
        buffers.release(i)

    def worker():
        for _ in range(50):
            i = acquire_tracked()
            assert buffers.canvas(i).shape == (64, 64, 3)
            release_tracked(i)

    ts = [threading.Thread(target=worker) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors


def test_pipeline_result_independent_of_thread_count():
    """三线程与串行的结果必须逐帧一致（并发不能改变内容）。"""
    outs = {}
    for mode in ("serial", "triple"):
        buffers = FrameBuffers(n=4, size=64)
        got = []

        def on_result(idx, dets, lat):
            got.append((idx, float(dets[0, 4])))

        pipe = Pipeline(FakeEngine(delay=0.001), buffers, make_source(25),
                        mode=mode, queue_size=4,
                        stages={"pre": fake_pre, "post": fake_post},
                        on_result=on_result)
        pipe.run()
        outs[mode] = sorted(got)

    assert outs["serial"] == outs["triple"], "流水线结果与串行不一致"
