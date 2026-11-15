# -*- coding: utf-8 -*-
"""优雅退出与异常恢复。"""
import threading
import time

import pytest

from conftest import FakeEngine, fake_post, fake_pre, make_source
from pipeline.buffers import FrameBuffers
from pipeline.pipeline import Pipeline


def test_all_threads_join_after_finish():
    """正常跑完后，三条线程都能 join 上（不是 daemon 强杀）。"""
    buffers = FrameBuffers(n=4, size=64)
    pipe = Pipeline(FakeEngine(delay=0.001), buffers, make_source(15),
                    mode="triple", queue_size=4,
                    stages={"pre": fake_pre, "post": fake_post})
    pipe.run()
    alive = [t.name for t in pipe._threads if t.is_alive()]
    assert not alive, "线程没退出：%s" % alive


def test_stop_event_exits_quickly():
    """发出停止信号后，管线应在 2 秒内收尾，而不是卡在阻塞队列上。"""
    buffers = FrameBuffers(n=4, size=64)

    def forever():
        i = 0
        while True:
            yield i, __import__("numpy").zeros((48, 64, 3), dtype="uint8")
            i += 1

    pipe = Pipeline(FakeEngine(delay=0.02), buffers, forever(),
                    mode="triple", queue_size=4,
                    stages={"pre": fake_pre, "post": fake_post})
    t = threading.Thread(target=pipe.run, daemon=True)
    t.start()
    time.sleep(0.5)
    pipe.stop.set()
    t.join(timeout=3)
    assert not t.is_alive(), "stop 之后 3 秒还没退出"


def test_stage_exception_propagates_and_stops_others():
    """某一级抛异常：其余线程要退出，主线程要把异常重新抛出来。"""
    buffers = FrameBuffers(n=4, size=64)

    def boom(raw, meta):
        raise RuntimeError("后处理炸了")

    pipe = Pipeline(FakeEngine(delay=0.001), buffers, make_source(200),
                    mode="triple", queue_size=4,
                    stages={"pre": fake_pre, "post": boom})
    with pytest.raises(RuntimeError, match="后处理炸了"):
        pipe.run()
    alive = [t.name for t in pipe._threads if t.is_alive()]
    assert not alive, "异常后仍有线程在跑：%s" % alive


def test_engine_failure_is_skipped_not_fatal():
    """引擎偶发返回 None：跳过该帧，管线继续跑完其余帧。"""
    buffers = FrameBuffers(n=4, size=64)
    pipe = Pipeline(FakeEngine(delay=0.001, fail_every=5), buffers,
                    make_source(20), mode="triple", queue_size=4,
                    stages={"pre": fake_pre, "post": fake_post})
    snap = pipe.run()
    # 20 帧里每 5 帧失一次 -> 约 16 帧成功，但不应该抛异常
    assert 10 <= snap["frames_out"] <= 20
    assert snap["frames_in"] == 20
