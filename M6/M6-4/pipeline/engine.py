# -*- coding: utf-8 -*-
"""推理引擎封装：单例 + 锁，并说明这里的线程安全边界。

结论先说：
    * onnxruntime 的 `InferenceSession.run` **本身是线程安全的**（内部有锁/线程池），
      多线程并发调用不会崩，只是不会更快。
    * 但 `TensorRTExecutionProvider` 的 `ExecutionContext` **不可多线程共享**，
      RKNN 的 context 同理。为了「换后端不改调用方」，这里统一用一把锁串行化。
    * 锁的粒度就是「一次 run」——不能更大，否则等于串行；也不能更小，
      否则后端切换时会踩内存。

为什么流水线仍然能加速（题面 §2.3 追问）：
    GIL 只保证「同一时刻只有一个线程执行 Python 字节码」。`session.run()`
    是 C++ 实现，调用期间会**释放 GIL**；`cv2.resize/cvtColor` 也是。
    所以预处理在算的时候，另一个线程可以真的在跑推理。
    纯 Python 的死循环（比如手写 NMS 的 while）才受 GIL 限制——
    这也是本项目的后处理虽在 Python 里、却仍能与推理重叠的原因：
    它短（1~2 ms），且大部分时间在 numpy 的 C 实现里。
"""
import os
import threading


class Engine:
    """线程安全的推理引擎。"""

    def __init__(self, model_path, providers=("CPUExecutionProvider",),
                 intra_threads=2):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=list(providers))
        self.in_name = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]
        self.model_path = model_path
        self._lock = threading.Lock()
        self._n_run = 0
        self._n_fail = 0

    def run(self, blob):
        """跑一次推理。失败返回 None，由流水线决定重试还是跳过。

        为什么用锁而不是依赖 Session 自身线程安全：
            一旦换成 TensorRT / RKNN 后端，共享 context 会直接崩。
            锁的代价在流水线里可以忽略（同一时刻只有一个推理线程）。
        """
        with self._lock:
            try:
                out = self.sess.run(None, {self.in_name: blob})[0]
                self._n_run += 1
                return out
            except Exception:
                self._n_fail += 1
                return None

    def stats(self):
        return {"runs": self._n_run, "failures": self._n_fail,
                "provider": self.provider,
                "model": os.path.basename(self.model_path)}


# 单例：整个进程只建一次会话，避免每个线程各自加载一份模型
_ENGINES = {}
_ENGINE_LOCK = threading.Lock()


def get_engine(model_path, providers=("CPUExecutionProvider",), intra_threads=2):
    key = (os.path.abspath(model_path), tuple(providers), intra_threads)
    with _ENGINE_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = Engine(model_path, providers, intra_threads)
        return _ENGINES[key]
