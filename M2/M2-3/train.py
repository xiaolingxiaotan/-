"""M2-3 训练主程序：数据 → 划分 → 建模 → 训练 → 导出 → 检测。

整条流水线拆成八个类，一个类包住一个部分，全部已实现::

    1 DataImporter    读标注 → 三种光照扩充 → 等比缩放填充 → 样本集
    2 DatasetSplitter 类别分层 + 场景整组 7:2:1 + 泄漏自检 + 写清单
    3 ModelBuilder    主干（四支路卷积）+ 三级 anchor-free 检测头
    4 LossFunction    CIoU 框回归 + 置信度 BCE + 分类 BCE
    5 OptimizerBuilder AdamW + 预热 + 余弦 + 梯度裁剪
    6 TrainingLoop    逐轮迭代、验证、早停、留档
    7 OutputExporter  权重、ONNX、曲线、配置、一致性验证
    8 Detector        用训练好的模型检测图片 / 视频 / 摄像头

依据：题目 ``一轮考核题目/22SmartCar/round1/M2/M2-3/README.md``
§3 数据与标注、§4 训练、§6 CPU 部署、§9 交付物、§10 明确禁止。

数据流（main 里的串联顺序也只是示意）::

    DataImporter       读 annotations.json -> 三种光照增强（暗光/背光/强光）
                       -> 等比缩放 + 居中填充到 size×size -> 样本集 (X, Y)
                       source: nature / dark / backlight / glare，按 group 归组
           |
           v
    DatasetSplitter    按「类别分层 + 场景整组」7:2:1 划分，产出 data/splits 清单
           |
           +----> ModelBuilder      搭主干 + 检测头
           +----> LossFunction      分类 / 框回归 / 置信度
           +----> OptimizerBuilder  优化器 + 学习率策略
           |
           v
    TrainingLoop       epoch 迭代、验证、迭代终止条件、训练留档
           |
           v
    OutputExporter     权重、ONNX、loss 曲线、日志、配置、一致性验证

写实现时别踩的硬约束（题面原文要点，先记在这里）::

    * 恰好 6 类，ID 顺序 = classes.txt：
      0 书 / 1 笔 / 2 杯子 / 3 保温杯 / 4 吧唧 / 5 草稿本；
      必须与 M2-4 的 detections JSON 里 ``classes`` 数组逐字一致。
    * 数据自采自标；原始采集数如实报告，不足才用同分布增强补足。
    * 每类 >=120 张有效图（总 >=720 张），类别数量分布要画柱状图。
    * 划分按场景：同一批次同背景的图不许跨 train/val/test（信息泄漏）。
    * val 集只用于调参；test 集只在最终报告时用一次，禁止用 test mAP 选模型。
    * 训练与 ONNX 部署的预处理必须严格一致（resize / 归一化 / 通道顺序）。
    * 现场无 GPU：模型与算子必须能在 CPU 上推理，实时 >=10 FPS。
    * 用预训练权重必须写明来源与 license；训练记录（loss 曲线、学习率
      策略、epoch 数、增强方式）全部留档，并给出一键训练命令。

本文件刻意不 import 任何深度学习框架：用 PyTorch / TensorFlow / PaddlePaddle
以及具体算子，都留给实现类自己决定。
"""

import hashlib
import json
import math
import os
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. 处理文件类：函数划分由用户自定，只保留下面三个
# ---------------------------------------------------------------------------
class DataImporter:
    """处理文件与预处理：读标注 → 三种光照扩充 → 等比缩放填充 → 输出样本集。

    这个类只做预处理，不含任何模型、损失、优化器。输出 (X, Y) 就是交给
    DatasetSplitter 的样本集。

    样本 dict 的字段：
        "image"   numpy uint8 (size, size, 3) RGB，letterbox 之后的正方形图；
        "source"  "nature" | "dark" | "backlight" | "glare"，来源标记；
        "group"   原始图的序号 i，同一张图的 4 个版本 group 相同。
                  按场景划分时必须整组进同一个 split，否则同一张图的不同
                  光照版会同时出现在 train 和 test 里，mAP 虚高；
        "geom"    {"scale", "pad_x", "pad_y", "orig_hw", "size"}，
                  letterbox 用的缩放比与填充量，评测时按
                  x_orig = (x - pad_x) / scale 反算回原图坐标；
        "stops"   dark / backlight / glare 额外带，记录该样本用了多少档。

    标签 Y[i] 是这张图的框列表 [x1, y1, x2, y2, class_id]，坐标已同步变换到
    letterbox 后的坐标系；X[i] 与 Y[i] 一一对应。

    三种光照的排列顺序是「按组连续」：n0 d0 b0 g0 n1 d1 b1 g1 ...，
    不是「全部原图 + 全部暗光」。这样下游按 group 切分不用再排序。

    刻意不做的事：
        * 不做任何滤波（高斯/锐化/中值是线性或会抹细节，网络第一层能自己学，
          唯一需要的抗混叠由 letterbox 选用的面积平均插值提供）；
        * 不做归一化，样本仍是 uint8。归一化放到取 batch 时做，省 4 倍内存，
          也便于保证训练与 ONNX 推理用同一套减均值除方差。
    """

    # ------------------------------------------------------------------
    # 0. 初始化
    # ------------------------------------------------------------------
    def __init__(self, input_size: int = 640, seed: int = 0,
                 pad_value: int = 114, augment: bool = True) -> None:
        """配置预处理参数，不改动任何数据。

        参数:
            input_size  letterbox 之后的正方形边长，默认 640。
                        它决定最细的目标还剩几像素：现场照片短边约 1280，
                        缩到 640 是 0.5 倍，笔的短边约 7~8 px，再小就学不出来。
                        改这个值必须连推理端一起改。
            seed        随机种子，固定后三种光照增强逐像素可复现，默认 0。
            pad_value   填充像素值，默认 114（YOLO 惯例的灰边）。
                        训练、ONNX 推理、现场部署必须用同一个值。
            augment     是否生成三种光照样本，默认 True。
                        False 时只做读图 + letterbox，样本数不翻倍。

        属性:
            input_size / seed / pad_value / augment
            dropped_boxes  上一次缩放后被丢弃的退化框数（缩放后短边 < 1 px）
        """
        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise TypeError("input_size 必须是整数，收到 %r" % (input_size,))
        if not 32 <= input_size <= 4096:
            raise ValueError("input_size 要在 [32, 4096] 内，收到 %d" % input_size)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed 必须是整数，收到 %r" % (seed,))
        if seed < 0:
            raise ValueError("seed 不能为负，收到 %d" % seed)
        if isinstance(pad_value, bool) or not isinstance(pad_value, int):
            raise TypeError("pad_value 必须是整数，收到 %r" % (pad_value,))
        if not 0 <= pad_value <= 255:
            raise ValueError("pad_value 要在 [0, 255] 内，收到 %d" % pad_value)

        self.input_size = int(input_size)
        self.seed = int(seed)
        self.pad_value = int(pad_value)
        self.augment = bool(augment)
        self.dropped_boxes = 0

    def load_annotations(self, json_path: str) -> Tuple[Any, Any]:
        """读 annotations.json，整理成输入 X（原图）与标签 Y（带标注的）。

        X：样本列表，每项 {"image": uint8 (H,W,3) RGB, "source": "nature"}，
           原始采集图统一打 nature 标记（暗光样本由 simulate_dark 生成）；
        Y：标注列表，每张是该图所有框组成的列表 [x1,y1,x2,y2,class_id]。
        返回 (X, Y)；json 路径由 --pictures 指定。

        找图的顺序：json 内嵌的 image.data -> data/raw/<file> ->
        image_path -> image.path -> json 同目录的 <file>。
        坏数据（框坐标反了、类别 ID 越界、图找不到、尺寸对不上）直接报错，
        不静默跳过任何一张。

        需要的库：numpy、Pillow（标注器已经用了这两个，不是深度学习框架）。
        """
        import json
        import os

        import numpy as np
        from PIL import Image, ImageOps

        if not os.path.isfile(json_path):
            raise FileNotFoundError("找不到标注文件：%s" % json_path)

        with open(json_path, encoding="utf-8") as f:
            doc = json.load(f)

        items = doc.get("items", [])
        if not isinstance(items, list):
            raise ValueError("annotations.json 里 items 必须是列表")

        class_ids = doc.get("class_ids")
        valid_ids = set(int(c) for c in class_ids) if class_ids else set(range(6))
        json_dir = os.path.dirname(os.path.abspath(json_path))
        raw_dir = os.path.join(json_dir, "data", "raw")

        X: List[Any] = []
        Y: List[Any] = []

        for i, item in enumerate(items):
            name = item.get("file")
            if not name:
                raise ValueError("第 %d 条标注缺 file 字段，无法确定图片名" % i)

            # ---- 标签 Y：逐个框检查 ----
            boxes = []
            for j, b in enumerate(item.get("boxes", [])):
                if not isinstance(b, (list, tuple)) or len(b) != 5:
                    raise ValueError(
                        "%s 第 %d 个框不是 [x1,y1,x2,y2,class_id]" % (name, j))
                x1, y1, x2, y2, cid = (float(b[0]), float(b[1]), float(b[2]),
                                       float(b[3]), float(b[4]))
                if not (x1 < x2 and y1 < y2):
                    raise ValueError("%s 第 %d 个框坐标反了：%s" % (name, j, b))
                if cid != int(cid) or int(cid) not in valid_ids:
                    raise ValueError(
                        "%s 第 %d 个框类别 ID 非法：%s" % (name, j, b[4]))
                boxes.append([x1, y1, x2, y2, int(cid)])

            # ---- 输入 X：优先用 json 内嵌像素，否则从磁盘读原图 ----
            image_info = item.get("image") or {}
            embedded = image_info.get("data")
            img = None
            if embedded is not None:
                img = np.ascontiguousarray(np.asarray(embedded, dtype=np.uint8))
            else:
                tried = []
                candidates = [
                    os.path.join(raw_dir, name),
                    item.get("image_path"),
                    image_info.get("path"),
                    os.path.join(json_dir, name),
                ]
                for p in candidates:
                    if not p:
                        continue
                    tried.append(p)
                    if os.path.isfile(p):
                        with Image.open(p) as im:
                            img = np.ascontiguousarray(
                                np.asarray(ImageOps.exif_transpose(im).convert("RGB"),
                                           dtype=np.uint8))
                        break
                if img is None:
                    raise FileNotFoundError(
                        "%s 的原图没找到，找过这些路径：\n  %s"
                        % (name, "\n  ".join(tried)))

            # 标注器按 EXIF 转正后的尺寸标框，尺寸对不上说明图被换过
            shape = image_info.get("shape")
            if shape and list(img.shape[:2]) != [int(shape[0]), int(shape[1])]:
                raise ValueError(
                    "%s 图片尺寸与标注时不一致：现在是 %s，标注时是 %s"
                    % (name, list(img.shape[:2]), list(shape[:2])))

            X.append({"image": img, "source": "nature", "file": name})
            Y.append(boxes)

        return X, Y

    def parse_args(self, argv: Optional[Sequence[str]] = None) -> Any:
        """读命令行指令，同时支持位置参数和 -- 参数。

            python train.py --pictures annotations.json
            python train.py -p annotations.json
            python train.py annotations.json      （位置参数，等价）
            python train.py --seed 123            （固定光照增强的随机种子）
            python train.py --size 512            （letterbox 边长，默认 640）
            python train.py --no-augment          （只做 resize，不生成光照样本）
            什么都不写 -> 默认读本文件同目录的 annotations.json，seed=0，size=640。

        优先级：--pictures > 位置参数 > 默认值。
        返回的 Namespace 带 .pictures / .seed / .size / .no_augment。
        注意 .no_augment 是布尔，构造 DataImporter 时要取反：
        ``DataImporter(input_size=args.size, seed=args.seed,
        augment=not args.no_augment)``。
        """
        import argparse
        import os

        parser = argparse.ArgumentParser(
            prog="train.py",
            description="M2-3 训练入口：第一个类（读标注 + 光照扩充 + letterbox）已实现。",
        )
        parser.add_argument("positional", nargs="?", default=None,
                            help="位置参数：annotations.json 的路径")
        parser.add_argument("-p", "--pictures", dest="pictures", default=None,
                            help="--pictures <json路径>：指定要读的 annotations.json")
        parser.add_argument("--seed", type=int, default=0,
                            help="随机种子：固定后光照增强结果可复现（默认 0）")
        parser.add_argument("--size", type=int, default=640,
                            help="letterbox 之后的边长，默认 640；改它要同步改推理端")
        parser.add_argument("--no-augment", dest="no_augment", action="store_true",
                            help="只做读图 + letterbox，不生成三种光照样本")
        parser.add_argument("--epochs", type=int, default=150,
                            help="最多训练多少轮（默认 150）")
        parser.add_argument("--batch", type=int, default=16,
                            help="批大小（默认 16）")
        parser.add_argument("--lr", type=float, default=1e-3,
                            help="初始学习率（默认 1e-3，AdamW 的量级）")
        parser.add_argument("--no-train", dest="no_train", action="store_true",
                            help="只做数据准备，不训练（调数据用）")
        parser.add_argument("--no-export", dest="no_export", action="store_true",
                            help="训练完不导出 ONNX（只想看指标时用）")
        args = parser.parse_args(argv)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        chosen = (args.pictures or args.positional
                  or os.path.join(script_dir, "annotations.json"))
        if not os.path.isabs(chosen) and not os.path.isfile(chosen):
            near_script = os.path.join(script_dir, chosen)
            if os.path.isfile(near_script):
                chosen = near_script
        args.pictures = os.path.abspath(chosen)
        return args

    def simulate_dark(self, X: Any, Y: Any,
                      seed: int = 0, stops: float = 3.0,
                      sigma: float = 0.04) -> Tuple[Any, Any]:
        """整批模拟暗光：每张原始样本生成一张暗光图，原图保留。

        这是「单独调一种光照、在原始分辨率上看效果」的入口。正式产样本集请用
        build_dataset()，它是逐张处理并及时释放原图的，内存只占这个整批版的
        四分之一左右。

        输入 X / Y 来自 load_annotations()。输出 (X2, Y2)：原样本原样保留在前，
        新增的 dark 样本按索引追加在后（第 i 张的暗光版在 i + N 位）。
        **只对 source == "nature" 的样本生成**：三种增强因此互不叠加，任意顺序
        串起来最终是 4N（原始 + 三种各 N）。但同一个函数别重复调用，否则会再加一份
        同类型增强。

        参数:
            seed   随机种子，同一个 seed 下噪声逐像素完全可复现；
            stops  暗几档，1 档 = 亮度减半，默认 3 档（约 1/8）；
            sigma  噪声强度，默认 0.04，越大越像弱光手机拍的。

        做法（sRGB 近似 gamma = 2.2）::

            L   = (img/255)^2.2                 反 gamma 到线性域
            L   = L * 2^(-stops)                减曝光（物理上等价于缩短曝光时间）
            L   = L + N(0, sigma^2 * L)         散粒噪声，标准差 ∝ 根号信号
            out = clip(255 * L^(1/2.2), 0, 255) 回 gamma、截断、转 uint8

        Y 不参与任何计算：光照是逐像素映射，不改变物体位置，框原样复制。
        """
        import numpy as np

        if len(X) != len(Y):
            raise ValueError("X 与 Y 长度不一致：%d vs %d" % (len(X), len(Y)))
        if not 0.0 < stops <= 4.0:
            raise ValueError("stops 要在 (0, 4] 内，太大会把暗部压成全黑：%s" % stops)
        if sigma < 0:
            raise ValueError("sigma 不能为负：%s" % sigma)

        rng = np.random.default_rng(int(seed))

        X_out = list(X)
        Y_out = [list(boxes) for boxes in Y]

        for i, sample in enumerate(X):
            if isinstance(sample, dict) and sample.get("source") != "nature":
                continue
            img = sample["image"] if isinstance(sample, dict) else sample

            dark = self._dark_one(img, rng, stops, sigma)

            X_out.append({"image": dark, "source": "dark",
                          "stops": float(stops)})
            Y_out.append([list(b) for b in Y[i]])

        return X_out, Y_out

    def simulate_backlight(self, X: Any, Y: Any,
                           seed: int = 0, stops: float = 1.5,
                           glow: float = 8.0, sigma_ratio: float = 0.18,
                           sigma: float = 0.03) -> Tuple[Any, Any]:
        """整批模拟背光（逆光）：光源在物体背后，背景过亮、主体欠曝成剪影。

        与 simulate_dark 同一契约：只对 source == "nature" 的样本生成，原样本保留，
        新增样本标 source="backlight"（额外带 "stops"），Y 原样复制。
        同样只是「单独调参看效果」的入口，产样本集请用 build_dataset()。

        参数:
            seed         随机种子，控制光源位置与噪声，可复现；
            stops        整幅欠曝档位，默认 1.5（相机以背后强光测光）；
            glow         光源中心相对亮度，默认 8.0（欠曝后背景仍然饱和成白）；
            sigma_ratio  光斑半径占图像短边的比例，默认 0.18（只覆盖上部背景）；
            sigma        噪声强度，默认 0.03（欠曝后暗部噪声更明显）。

        做法（线性域，光源场用可分离高斯算，省内存）::

            L = (img/255)^2.2
            G = 1 + glow * exp(-dx^2/(2s^2)) * exp(-dy^2/(2s^2))
            L = L * G * 2^(-stops)     先被背后光源照亮，再整体欠曝
            L = clip(L, 0, 1)          背景高光截断成白（不可逆）
            out = 255 * L^(1/2.2)

        Y 不动。
        """
        import numpy as np

        if len(X) != len(Y):
            raise ValueError("X 与 Y 长度不一致：%d vs %d" % (len(X), len(Y)))
        if not 0.0 <= stops <= 4.0:
            raise ValueError("stops 要在 [0, 4] 内：%s" % stops)
        if glow < 0:
            raise ValueError("glow 不能为负：%s" % glow)
        if not 0.0 < sigma_ratio <= 1.0:
            raise ValueError("sigma_ratio 要在 (0, 1] 内：%s" % sigma_ratio)
        if sigma < 0:
            raise ValueError("sigma 不能为负：%s" % sigma)

        gamma = 2.2
        alpha = float(2.0 ** (-float(stops)))
        rng = np.random.default_rng(int(seed))

        X_out = list(X)
        Y_out = [list(boxes) for boxes in Y]

        for i, sample in enumerate(X):
            if not isinstance(sample, dict) or sample.get("source") != "nature":
                continue
            img = sample["image"]

            out = self._backlight_one(img, rng, stops, glow, sigma_ratio, sigma)

            X_out.append({"image": out, "source": "backlight",
                          "stops": float(stops)})
            Y_out.append([list(b) for b in Y[i]])

        return X_out, Y_out

    def simulate_glare(self, X: Any, Y: Any,
                       seed: int = 0, stops: float = 0.35,
                       glow: float = 4.0, veil: float = 0.10,
                       sigma_ratio: float = 0.18,
                       sigma: float = 0.02) -> Tuple[Any, Any]:
        """整批模拟强光直射：整体过曝、光源处高光溢出、对比度被眩光抬高。

        契约同前两个：只对 source == "nature" 的样本生成，原样本保留，
        新增样本标 source="glare"（额外带 "stops"），Y 原样复制。
        同样只是「单独调参看效果」的入口，产样本集请用 build_dataset()。

        参数:
            seed         随机种子，控制光源位置与噪声，可复现；
            stops        整体过曝档位，默认 0.35（光斑之外只轻微过曝）；
            glow         直射光源相对亮度，默认 4.0（光斑内饱和成白）；
            veil         眩光雾化强度，默认 0.10（抬高黑位、压低对比度）；
            sigma_ratio  光斑半径占短边比例，默认 0.18（一小块强光）；
            sigma        噪声强度，默认 0.02。

        做法::

            L = (img/255)^2.2
            L = L * (1 + glow * gap)            直射光斑
            L = L * 2^(+stops)                  整体过曝
            L = L * (1 - veil) + veil * 0.02    眩光抬黑位（veiling glare）
            L = clip(L, 0, 1)                   高光溢出成白（不可逆）
            out = 255 * L^(1/2.2)

        Y 不动。
        """
        import numpy as np

        if len(X) != len(Y):
            raise ValueError("X 与 Y 长度不一致：%d vs %d" % (len(X), len(Y)))
        if not 0.0 <= stops <= 4.0:
            raise ValueError("stops 要在 [0, 4] 内：%s" % stops)
        if glow < 0:
            raise ValueError("glow 不能为负：%s" % glow)
        if not 0.0 <= veil < 1.0:
            raise ValueError("veil 要在 [0, 1) 内：%s" % veil)
        if not 0.0 < sigma_ratio <= 1.0:
            raise ValueError("sigma_ratio 要在 (0, 1] 内：%s" % sigma_ratio)
        if sigma < 0:
            raise ValueError("sigma 不能为负：%s" % sigma)

        gamma = 2.2
        rng = np.random.default_rng(int(seed))

        X_out = list(X)
        Y_out = [list(boxes) for boxes in Y]

        for i, sample in enumerate(X):
            if not isinstance(sample, dict) or sample.get("source") != "nature":
                continue
            img = sample["image"]

            out = self._glare_one(img, rng, stops, glow, veil,
                                  sigma_ratio, sigma)

            X_out.append({"image": out, "source": "glare",
                          "stops": float(stops)})
            Y_out.append([list(b) for b in Y[i]])

        return X_out, Y_out

    # ------------------------------------------------------------------
    # 3b. 三种光照的单张实现（内部函数，上面三个整批版与 build_dataset 共用）
    #     抽出来的原因：整批版一次把 4N 张原分辨率图留在内存里会爆，
    #     build_dataset 必须逐张处理，两处不能各写一套，否则参数会走偏。
    # ------------------------------------------------------------------
    @staticmethod
    def _check_image(img: Any, where: str = "") -> None:
        """确认是 uint8 (H, W, 3)。所有入口共用，避免 dtype 悄悄变 float。"""
        import numpy as np

        tag = ("第 %s 张" % where) if where else "图像"
        if not isinstance(img, np.ndarray):
            raise TypeError("%s不是 numpy 数组，而是 %s" % (tag, type(img).__name__))
        if img.dtype != np.uint8:
            raise ValueError("%s的 dtype 是 %s，必须是 uint8" % (tag, img.dtype))
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("%s的形状是 %s，必须是 (H, W, 3) RGB"
                             % (tag, img.shape))
        if img.shape[0] < 1 or img.shape[1] < 1:
            raise ValueError("%s尺寸为 0：%s" % (tag, img.shape))

    @staticmethod
    def _dark_one(img: Any, rng: Any, stops: float, sigma: float) -> Any:
        """单张暗光：反 gamma → 减曝光 → 加散粒噪声 → 回 gamma。Y 无关。"""
        import numpy as np

        DataImporter._check_image(img)
        gamma = 2.2
        alpha = float(2.0 ** (-float(stops)))

        lin = np.power(img.astype(np.float32) / 255.0, gamma)
        lin *= alpha
        noise = rng.standard_normal(lin.shape).astype(np.float32)
        lin = lin + noise * (float(sigma) * np.sqrt(lin + 1e-6))
        lin = np.clip(lin, 0.0, 1.0)
        out = np.clip(np.power(lin, 1.0 / gamma) * 255.0, 0.0, 255.0)
        return out.astype(np.uint8)

    @staticmethod
    def _backlight_one(img: Any, rng: Any, stops: float, glow: float,
                       sigma_ratio: float, sigma: float) -> Any:
        """单张背光：背后一小块高亮 + 整幅欠曝，背景截断成白、主体成剪影。"""
        import numpy as np

        DataImporter._check_image(img)
        gamma = 2.2
        alpha = float(2.0 ** (-float(stops)))

        h, w = img.shape[:2]
        lin = np.power(img.astype(np.float32) / 255.0, gamma)

        # 光源在画面上部（背后的窗户/灯），光斑只覆盖背景那一块
        cx = float(rng.uniform(0.30, 0.70)) * w
        cy = float(rng.uniform(0.05, 0.20)) * h
        s = float(sigma_ratio) * min(h, w)
        s2 = 2.0 * s * s
        gx = np.exp(-((np.arange(w, dtype=np.float32) - cx) ** 2) / s2)
        gy = np.exp(-((np.arange(h, dtype=np.float32) - cy) ** 2) / s2)
        gain = (1.0 + float(glow) * np.outer(gy, gx)).astype(np.float32)

        lin = lin * gain[:, :, None]
        lin = lin * alpha
        noise = rng.standard_normal(lin.shape).astype(np.float32)
        lin = lin + noise * (float(sigma) * np.sqrt(lin + 1e-6))
        lin = np.clip(lin, 0.0, 1.0)
        out = np.clip(np.power(lin, 1.0 / gamma) * 255.0, 0.0, 255.0)
        return out.astype(np.uint8)

    @staticmethod
    def _glare_one(img: Any, rng: Any, stops: float, glow: float,
                   veil: float, sigma_ratio: float, sigma: float) -> Any:
        """单张强光直射：直射光斑 + 轻微过曝 + 眩光抬黑位（发灰）。"""
        import numpy as np

        DataImporter._check_image(img)
        gamma = 2.2
        alpha = float(2.0 ** float(stops))

        h, w = img.shape[:2]
        lin = np.power(img.astype(np.float32) / 255.0, gamma)

        # 直射光源位置更随意，可能在画面任意一侧
        cx = float(rng.uniform(0.20, 0.80)) * w
        cy = float(rng.uniform(0.20, 0.80)) * h
        s = float(sigma_ratio) * min(h, w)
        s2 = 2.0 * s * s
        gx = np.exp(-((np.arange(w, dtype=np.float32) - cx) ** 2) / s2)
        gy = np.exp(-((np.arange(h, dtype=np.float32) - cy) ** 2) / s2)
        gain = (1.0 + float(glow) * np.outer(gy, gx)).astype(np.float32)

        lin = lin * gain[:, :, None]
        lin = lin * alpha
        lin = lin * (1.0 - float(veil)) + float(veil) * 0.02
        noise = rng.standard_normal(lin.shape).astype(np.float32)
        lin = lin + noise * (float(sigma) * np.sqrt(lin + 1e-6))
        lin = np.clip(lin, 0.0, 1.0)
        out = np.clip(np.power(lin, 1.0 / gamma) * 255.0, 0.0, 255.0)
        return out.astype(np.uint8)

    # ------------------------------------------------------------------
    # 4. 等比缩放 + 填充（reshape）
    # ------------------------------------------------------------------
    @staticmethod
    def _check_size(size: Any) -> int:
        """校验 letterbox 边长。"""
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("size 必须是整数，收到 %r" % (size,))
        if not 32 <= size <= 4096:
            raise ValueError("size 要在 [32, 4096] 内，收到 %d" % size)
        return int(size)

    @staticmethod
    def _letterbox_one(img: Any, boxes: Any, size: int,
                       pad_value: int) -> Tuple[Any, Any, Any, int]:
        """单张等比缩放 + 居中填充，框坐标同步变换。

        返回 (新图, 新框列表, geom, 丢弃的退化框数)。

        插值方式就是那个「抗混叠」的唯一落点：
            * 缩小 -> 面积平均（Image.BOX），等价于先低通再抽样，
              不会把高频折叠成低频；这是唯一必须的低通，不做就丢信息且不可逆；
            * 放大 -> 双三次（Image.BICUBIC）。
        除此之外不加任何滤波。

        几何约定：原图 (w, h) -> 等比 r = min(size/w, size/h) -> (nw, nh)
        居中贴进 size×size 的画布，左上角偏移 (left, top)，
        所以 x_new = x * r + left，反算 x = (x_new - left) / r。

        缩放后短边不足 1 px 的框会被丢弃并计数（保留会得到退化框，
        后面算 IoU 和损失时是 NaN 的来源）。丢弃数记在 self.dropped_boxes。
        """
        import numpy as np
        from PIL import Image

        DataImporter._check_image(img)
        size = DataImporter._check_size(int(size))
        if not 0 <= int(pad_value) <= 255:
            raise ValueError("pad_value 要在 [0, 255] 内，收到 %r" % (pad_value,))

        h, w = img.shape[:2]
        r = min(size / float(w), size / float(h))
        nw = int(round(w * r))
        nh = int(round(h * r))
        if nw < 1 or nh < 1:
            raise ValueError(
                "原图 %dx%d 缩到 %d 后某一边不足 1 像素，size 设得太小"
                % (w, h, size))

        res = getattr(Image, "Resampling", Image)
        interp = res.BOX if r < 1.0 else res.BICUBIC
        with Image.fromarray(img) as im:
            small = np.asarray(im.resize((nw, nh), interp), dtype=np.uint8)

        canvas = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = small

        out_boxes = []
        dropped = 0
        for j, b in enumerate(boxes):
            if not isinstance(b, (list, tuple)) or len(b) != 5:
                raise ValueError("第 %d 个框不是 [x1,y1,x2,y2,class_id]：%r"
                                 % (j, b))
            nx1 = min(max(float(b[0]) * r + left, 0.0), float(size))
            ny1 = min(max(float(b[1]) * r + top, 0.0), float(size))
            nx2 = min(max(float(b[2]) * r + left, 0.0), float(size))
            ny2 = min(max(float(b[3]) * r + top, 0.0), float(size))
            if nx2 - nx1 < 1.0 or ny2 - ny1 < 1.0:
                dropped += 1
                continue
            out_boxes.append([nx1, ny1, nx2, ny2, int(b[4])])

        geom = {"scale": float(r), "pad_x": int(left), "pad_y": int(top),
                "orig_hw": (int(h), int(w)), "size": int(size)}
        return canvas, out_boxes, geom, dropped

    def letterbox(self, X: Any, Y: Any, size: Optional[int] = None) -> Tuple[Any, Any]:
        """整批 reshape：等比缩放 + 居中填充，框同步变换。

        输入是 load_annotations() 出来的原分辨率样本，输出是 size×size 的样本集。
        每个样本会多一个 "geom" 字段，记录缩放比与填充量，评测时按
        x_orig = (x - pad_x) / scale 反算回原图。

        正式产样本集请用 build_dataset()（它会顺手做光照扩充、并及时释放原图）；
        这个函数是给「只想 resize、不要增强」的场景单独用的。

        异常：X/Y 长度不一致、样本不是 dict、样本已带 geom（重复调用会叠加灰边）。
        """
        if len(X) != len(Y):
            raise ValueError("X 与 Y 长度不一致：%d vs %d" % (len(X), len(Y)))
        size = self.input_size if size is None else self._check_size(size)

        X_out: List[Any] = []
        Y_out: List[Any] = []
        dropped = 0
        for i, sample in enumerate(X):
            if not isinstance(sample, dict) or "image" not in sample:
                raise ValueError(
                    "第 %d 个样本不是 {'image': ...} 字典，而是 %s"
                    % (i, type(sample).__name__))
            if "geom" in sample:
                raise ValueError(
                    "第 %d 个样本已经 letterbox 过（带 geom），再调一次会重复"
                    "加灰边。要么直接用原分辨率样本，要么别重复调用。" % i)
            img, boxes, geom, nd = self._letterbox_one(
                sample["image"], Y[i], size, self.pad_value)
            dropped += nd
            rec = dict(sample)
            rec["image"] = img
            rec["geom"] = geom
            X_out.append(rec)
            Y_out.append(boxes)

        self.dropped_boxes = dropped
        return X_out, Y_out

    # ------------------------------------------------------------------
    # 5. 主链：读好的原图 -> 完整样本集（这个类的输出）
    # ------------------------------------------------------------------
    def build_dataset(self, X: Any, Y: Any, seed: Optional[int] = None,
                      augment: Optional[bool] = None,
                      size: Optional[int] = None) -> Tuple[Any, Any]:
        """把 load_annotations() 的输出变成最终样本集，这是本类的输出产物。

        顺序就是题面要求的那条链::

            原图 + 标签
              -> 三种光照各生成一张（nature / dark / backlight / glare）
              -> 逐张等比缩放 + 填充到 size×size，框同步变换
              -> 样本集 (X, Y)

        与「三个 simulate_ 串起来再 letterbox」的区别只有内存：
        这里是**逐张**处理完就把原图引用置 None，峰值只有一份全量原图；
        整批版会同时持有 4N 张原分辨率图，1706×1279 每张 6.5 MB，
        377 张四份就是 9.6 GB，WSL 直接 OOM。

        参数:
            seed     随机种子，默认取 self.seed；每张图的 rng 由 (seed, i) 派生，
                     所以结果与处理顺序无关，单独重跑第 i 张也能复现；
            augment  是否生成三种光照，默认取 self.augment；
            size     letterbox 边长，默认取 self.input_size。

        返回 (X, Y)：
            顺序按组连续 n0 d0 b0 g0 n1 d1 b1 g1 ...
            样本字段见类 docstring（image / source / group / geom / stops）。

        注意（会改调用方的数据）：
            为了让内存真的降下来，处理完第 i 张会把 ``X[i]`` 置成 None。
            传进来的 X 必须是可以下标赋值的 list；如果你还要用原图，
            自己先拷一份再用 None 之外的方式释放。

        异常：
            X/Y 长度不一致、样本数为 0、source 不是 nature、
            单张光照增强失败（带图序号抛出，不静默跳过）。
        """
        import numpy as np

        if not isinstance(X, list) or not isinstance(Y, list):
            raise TypeError("X 与 Y 都必须是 list（load_annotations 的返回值）")
        if len(X) != len(Y):
            raise ValueError("X 与 Y 长度不一致：%d vs %d" % (len(X), len(Y)))
        if len(X) == 0:
            raise ValueError(
                "样本数为 0：annotations.json 里还没有标注。先用 label_tool.py "
                "标图，或者检查 --pictures 指对了没有。")

        seed = self.seed if seed is None else int(seed)
        augment = self.augment if augment is None else bool(augment)
        size = self.input_size if size is None else self._check_size(size)

        # 与三个 simulate_ 的默认值保持一致，改一处要三处一起看
        light_cfg = {
            "dark": dict(stops=3.0, sigma=0.04),
            "backlight": dict(stops=1.5, glow=8.0, sigma_ratio=0.18,
                              sigma=0.03),
            "glare": dict(stops=0.35, glow=4.0, veil=0.10,
                          sigma_ratio=0.18, sigma=0.02),
        }

        X_out: List[Any] = []
        Y_out: List[Any] = []
        dropped = 0
        n = len(X)

        for i in range(n):
            sample = X[i]
            if not isinstance(sample, dict) or "image" not in sample:
                raise ValueError("第 %d 个样本不是 {'image': ...} 字典" % i)
            src = sample.get("source", "nature")
            if src != "nature":
                raise ValueError(
                    "第 %d 个样本的 source 是 %r：build_dataset 只接受原始 "
                    "nature 样本，三种光照由它内部生成，别先手动增强一遍。"
                    % (i, src))
            img = sample["image"]
            self._check_image(img, where=str(i))

            # 每张图一个独立 rng：结果与处理顺序无关，单张也能复现
            rng = np.random.default_rng([int(seed), int(i)])

            variants = [("nature", img, {})]
            if augment:
                try:
                    variants.append(
                        ("dark",
                         self._dark_one(img, rng, **light_cfg["dark"]),
                         {"stops": light_cfg["dark"]["stops"]}))
                    variants.append(
                        ("backlight",
                         self._backlight_one(img, rng,
                                             **light_cfg["backlight"]),
                         {"stops": light_cfg["backlight"]["stops"]}))
                    variants.append(
                        ("glare",
                         self._glare_one(img, rng, **light_cfg["glare"]),
                         {"stops": light_cfg["glare"]["stops"]}))
                except (ValueError, TypeError) as exc:
                    raise RuntimeError(
                        "第 %d 张（%s）光照增强失败：%s"
                        % (i, sample.get("file", "未知文件名"), exc)) from exc

            for source, vimg, extra in variants:
                lb, boxes, geom, nd = self._letterbox_one(
                    vimg, Y[i], size, self.pad_value)
                dropped += nd
                rec = {"image": lb, "source": source, "group": i, "geom": geom,
                       "file": sample.get("file")}
                rec.update(extra)
                X_out.append(rec)
                Y_out.append(boxes)

            # 及时释放：否则 4N 张原分辨率图会同时留在内存里
            X[i] = None

        self.dropped_boxes = dropped
        return X_out, Y_out

    # ------------------------------------------------------------------
    # 6. 输出统计
    # ------------------------------------------------------------------
    @staticmethod
    def _load_class_names() -> Dict[int, str]:
        """读同目录 classes.txt 的类别名（跳过 # 注释行），读不到就只显示 ID。

        文件里一行一个 ``<id> <名字>``；注释行、空行、名字留空的行都跳过，
        不因为这些行让整张表失效。
        """
        import os

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "classes.txt")
        names: Dict[int, str] = {}
        if not os.path.isfile(path):
            return names
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) != 2:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    name = parts[1].strip()
                    if name:
                        names[cid] = name
        except OSError:
            return {}
        return names

    def summary(self, X: Any, Y: Any) -> Dict[str, Any]:
        """统计并打印样本集画像，返回同一个 dict（供后面的类/报告直接用）。

        报的东西：来源分布、每类框数、框尺寸的 sqrt(w·h) 分位数、
        短边 < 8 px 的框数（细长目标的风险）、丢弃的退化框数、内存占用。

        类别分布是题面 §3 要求画的柱状图的数据源，每类 >=120 张的判断也看这里。
        """
        import numpy as np

        if len(X) != len(Y):
            raise ValueError("X 与 Y 长度不一致：%d vs %d" % (len(X), len(Y)))
        if len(X) == 0:
            raise ValueError("样本集为空，没有可统计的东西")

        src: Dict[str, int] = {}
        cls: Dict[int, int] = {}
        areas: List[float] = []
        min_sides: List[float] = []
        thin = 0
        for i, sample in enumerate(X):
            if not isinstance(sample, dict) or "image" not in sample:
                raise ValueError("第 %d 个样本不是 {'image': ...} 字典" % i)
            s = sample.get("source", "?")
            src[s] = src.get(s, 0) + 1
            for b in Y[i]:
                bw = float(b[2]) - float(b[0])
                bh = float(b[3]) - float(b[1])
                short = min(bw, bh)
                if short <= 0:
                    raise ValueError("第 %d 张存在零面积框：%r" % (i, b))
                areas.append((bw * bh) ** 0.5)
                min_sides.append(short)
                if short < 8.0:
                    thin += 1
                cid = int(b[4])
                cls[cid] = cls.get(cid, 0) + 1

        names = self._load_class_names()
        a = np.asarray(areas, dtype=np.float64)
        m = np.asarray(min_sides, dtype=np.float64)
        size = X[0]["geom"]["size"] if X[0].get("geom") else self.input_size
        mem_gb = len(X) * size * size * 3 / (1024.0 ** 3)

        print("样本集：%d 张、%d 个框（size=%d, pad=%d, augment=%s）"
              % (len(X), len(areas), size, self.pad_value,
                 any(k != "nature" for k in src)))
        print("来源：" + " / ".join("%s %d" % (k, v) for k, v in sorted(src.items())))
        if cls:
            print("类别：" + " / ".join(
                "%s %d" % (names.get(c, "id=%d" % c), n)
                for c, n in sorted(cls.items())))
        if len(a):
            q = np.percentile(a, [0, 5, 50, 95, 100])
            print("框尺寸 sqrt(w*h)：min %.1f / P5 %.1f / P50 %.1f / P95 %.1f / "
                  "max %.1f" % tuple(q))
            print("短边 min(w,h)：min %.1f / P5 %.1f / P50 %.1f px，"
                  "短边 < 8 px 的框 %d 个（%.1f%%）"
                  % (m.min(), np.percentile(m, 5), np.median(m),
                     thin, 100.0 * thin / len(m)))
        print("缩放后短边不足 1 px 被丢弃：%d 个框" % self.dropped_boxes)
        print("内存占用（uint8 常驻）：%.2f GB" % mem_gb)

        return {"n_samples": len(X), "n_boxes": len(areas), "source": src,
                "class_counts": cls, "class_names": names,
                "box_side_p5": float(np.percentile(a, 5)) if len(a) else 0.0,
                "box_side_p95": float(np.percentile(a, 95)) if len(a) else 0.0,
                "thin_boxes": thin, "dropped_boxes": self.dropped_boxes,
                "size": size, "mem_gb": mem_gb}


# ---------------------------------------------------------------------------
# 2. 样本集划分类
# ---------------------------------------------------------------------------
class DatasetSplitter:
    """按「类别分层 + 场景整组」把样本集划成 train / val / test 7:2:1。

    分层解决的是类别不均衡
    ----------------------
    6 类很不均衡（书 110 组、杯子只有 25 组）。纯随机 7:2:1，杯子在 test 里
    期望只剩 2~3 组，mAP 抖动会大到没法写进报告。分层就是按类别配额分，
    让每类在三个集合里的比例都贴近 7:2:1，稀有类不会某个集合直接空掉。

    分层单位不是单张图，而是先合并好的 unit
    --------------------------------------
    三件必须整组走的事，任何一件跨集合都是泄漏：
        1) 同一张原图的 nature/dark/backlight/glare 四版（同 group）——
           等于同一张图同时进 train 和 test，mAP 会虚高十几个点；
        2) 同一场景批次的不同原图（同背景同光照）——模型会记住背景而不是物体；
        3) 内容完全相同的图（不同文件名、不同 group）——图库里有历史重复。
    所以先把 group 按「场景」和「内容哈希」并成 unit，unit 是最小不可分单元，
    整个 unit 只会落进一个集合。

    多标签怎么分层
    --------------
    一张图里既有书又有杯子，没法同时按两个类分层。做法是取该 unit 类别集合里
    **最稀有的那个类**作为分层键：先照顾稀有类，常见类因为覆盖面广，比例会
    自然跟着准。这是多标签分层的一种近似（严格做法是迭代分层，代价大很多），
    对本数据规模够用，实际偏差会如实打在 balance_report 里。
    """

    def __init__(self, config: Any = None,
                 ratio: Sequence[float] = (0.7, 0.2, 0.1),
                 seed: int = 0, scene_map: Any = None) -> None:
        """配置划分参数。

        参数:
            ratio      三个集合的比例，默认 (0.7, 0.2, 0.1)，必须都为正且和为 1；
            seed       随机种子，同 seed 划分结果完全可复现；
            scene_map  「文件名或 group 号 -> 场景标签」的映射。可以是 dict，也可以
                       是 json 路径，形如 {"IMG_0001.jpg": "桌面A-自然光"}。
                       **强烈建议提供**：不给时退化成「每张原图自己一个场景」，
                       这样 group 泄漏和重复图泄漏仍然防住，但同一批拍摄、同背景
                       同光照的不同原图会被拆到不同集合，场景级泄漏防不住，
                       test mAP 会偏高。balance_report 会把这条写进 notes。

        属性（split 之后才有值）:
            X / Y       本次划分用的样本集；
            units       合并后的最小不可分单元列表；
            splits      {'train': [...], 'val': [...], 'test': [...]}，元素是样本下标；
            assign      unit_id -> 'train' / 'val' / 'test'；
            scene_of    文件名 -> 场景标签。
        """
        if len(ratio) != 3:
            raise ValueError("ratio 必须是三个数（train, val, test），收到 %r"
                             % (ratio,))
        r = [float(x) for x in ratio]
        if any(x <= 0.0 or x >= 1.0 for x in r):
            raise ValueError("ratio 每一项都要在 (0, 1) 内，收到 %r" % (r,))
        if abs(sum(r) - 1.0) > 1e-6:
            raise ValueError("ratio 三项之和必须是 1，现在是 %.6f" % sum(r))
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed 必须是整数，收到 %r" % (seed,))

        self.config = config
        self.ratio = tuple(r)
        self.seed = int(seed)
        self.scene_map = self._load_scene_map(scene_map)

        self.X = None
        self.Y = None
        self.units = None
        self.splits = None
        self.assign = None
        self.scene_of = None
        self.splits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "data", "splits")

    # ------------------------------------------------------------------
    # 0. 场景映射的装载与判定
    # ------------------------------------------------------------------
    @staticmethod
    def _load_scene_map(scene_map: Any) -> Dict[str, str]:
        """把 dict 或 json 路径统一成 {key: 场景标签}；key 一律转成字符串。"""
        if scene_map is None:
            return {}
        if isinstance(scene_map, dict):
            return {str(k): str(v) for k, v in scene_map.items()}
        if isinstance(scene_map, str):
            if not os.path.isfile(scene_map):
                raise FileNotFoundError("场景映射文件不存在：%s" % scene_map)
            with open(scene_map, encoding="utf-8") as f:
                doc = json.load(f)
            if not isinstance(doc, dict):
                raise ValueError(
                    "场景映射 json 必须是 {\"文件名\": \"场景标签\"} 这样的对象，"
                    "实际是 %s" % type(doc).__name__)
            return {str(k): str(v) for k, v in doc.items()}
        raise TypeError("scene_map 只能是 dict、json 路径或 None，收到 %r"
                        % (type(scene_map).__name__,))

    def assign_scene(self, record: Any) -> str:
        """判定一个样本属于哪个场景批次。

        查找顺序：
            1) 样本自带 "scene" 字段（标注时写过就直接用）；
            2) scene_map 里按文件名 / 去扩展名的文件名 / group 号查；
            3) 都没有 -> "g<group 号>"，即「每张原图自己一个场景」。
                注意这不是「绝对安全」：group 泄漏与重复图泄漏仍然防住，但同一批
                拍摄、同背景同光照的不同原图会被当成不同场景拆开，场景级泄漏防不住。
                要真正按场景划分，必须提供 scene_map，见 balance_report 的 notes。
        """
        if not isinstance(record, dict):
            raise TypeError("assign_scene 需要样本 dict，收到 %s"
                            % type(record).__name__)

        scene = record.get("scene")
        if scene:
            return str(scene)

        name = record.get("file")
        if name and self.scene_map:
            base = os.path.basename(str(name))
            stem = os.path.splitext(base)[0]
            for key in (str(name), base, stem):
                if key in self.scene_map:
                    return self.scene_map[key]

        g = record.get("group")
        if g is not None and self.scene_map and str(g) in self.scene_map:
            return self.scene_map[str(g)]

        return "g%06d" % (int(g) if g is not None else -1)

    # ------------------------------------------------------------------
    # 1. 合并成最小不可分单元
    # ------------------------------------------------------------------
    @staticmethod
    def _content_key(X: Any, idxs: Any) -> str:
        """一张原图的内容指纹（形状 + 像素字节的 sha1）。

        用 nature 那一份算；同一张原图的四个光照版内容不同，但 group 相同，
        所以内容去重按 group 的代表图算就够了。
        """
        pick = None
        for i in idxs:
            if X[i].get("source") == "nature":
                pick = X[i]
                break
        if pick is None:
            pick = X[idxs[0]]
        img = pick.get("image")
        if img is None:
            raise ValueError("样本缺 image 字段，无法算内容指纹")
        h = hashlib.sha1()
        h.update(str(getattr(img, "shape", "")).encode())
        h.update(img.tobytes())
        return h.hexdigest()

    def group_by_scene(self, X: Any = None, Y: Any = None) -> List[Any]:
        """把样本并成 unit：同 group、同场景、同内容的一律并在一起。

        返回 unit 列表，每个 unit 是::

            {"uid": 整数, "groups": [...], "scenes": [...], "contents": [...],
             "classes": frozenset([...]), "indices": [样本下标...]}

        "indices" 就是这个 unit 覆盖的全部样本下标（含四种光照版）。
        """
        X, Y = self._resolve(X, Y)

        by_group: Dict[Any, List[int]] = {}
        for idx, s in enumerate(X):
            by_group.setdefault(s.get("group", idx), []).append(idx)
        gids = sorted(by_group, key=lambda g: (str(type(g)), g))
        pos = {g: k for k, g in enumerate(gids)}

        parent = list(range(len(gids)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        info: Dict[Any, Dict[str, Any]] = {}
        first_scene: Dict[str, Any] = {}
        first_content: Dict[str, str] = {}

        for g in gids:
            idxs = by_group[g]
            head = X[idxs[0]]
            scene = self.assign_scene(head)
            ckey = self._content_key(X, idxs)
            classes = frozenset(int(b[4]) for i in idxs for b in Y[i])
            info[g] = {"group": g, "scene": scene, "content": ckey,
                       "classes": classes, "indices": idxs}

            if scene in first_scene:
                union(pos[g], pos[first_scene[scene]])
            else:
                first_scene[scene] = g
            if ckey in first_content:
                union(pos[g], pos[first_content[ckey]])
            else:
                first_content[ckey] = g

        merged: Dict[int, List[Any]] = {}
        for g in gids:
            merged.setdefault(find(pos[g]), []).append(info[g])

        units: List[Any] = []
        for uid, root in enumerate(sorted(merged)):
            members = merged[root]
            units.append({
                "uid": uid,
                "groups": sorted(m["group"] for m in members),
                "scenes": sorted({m["scene"] for m in members}),
                "contents": sorted({m["content"] for m in members}),
                "classes": frozenset().union(*[m["classes"] for m in members]),
                "indices": sorted(i for m in members for i in m["indices"]),
            })

        if not units:
            raise ValueError("样本集为空，没有可划分的单元")

        self.units = units
        self.scene_of = {X[i].get("file"): self.assign_scene(X[i])
                         for i in range(len(X))}
        return units

    # ------------------------------------------------------------------
    # 2. 按类别分层分配
    # ------------------------------------------------------------------
    def _alloc(self, n: int) -> Tuple[int, int, int]:
        """把 n 个 unit 按 ratio 拆成三份，用最大余数法，保证加起来等于 n。

        全程整数运算：0.7 这类浮点数在 ``1.4 - 1 = 0.3999999999999999`` 上会翻车，
        让余数比较偏向 val，小桶就会莫名少给 train 一个 unit。
        """
        weight = [int(round(x * 10000)) for x in self.ratio]
        total = sum(weight)
        if total != 10000:
            raise ValueError("ratio 精度不足，无法转成整数权重：%r" % (self.ratio,))
        numer = [n * w for w in weight]
        base = [q // total for q in numer]
        rem = [q % total for q in numer]
        rest = n - sum(base)
        order = sorted(range(3), key=lambda i: (-rem[i], i))
        for i in order[:rest]:
            base[i] += 1

        # 小桶保护：n>=3 时 val 和 test 各至少 1 个。
        # 3 个 unit 按 7:2:1 取整是 2/1/0，test 恒空，稀有类的 test mAP 直接是 nan；
        # 大桶（>=10）本来就不会出现 0，这里只影响小桶，比例会偏均匀一些，
        # 但「三分都存在」比「严格 7:2:1 却缺一个集合」重要。
        if n >= 3:
            for idx in (1, 2):
                if base[idx] == 0:
                    donor = max(range(3), key=lambda i: base[i])
                    if base[donor] > 1:
                        base[donor] -= 1
                        base[idx] += 1
        return int(base[0]), int(base[1]), int(base[2])

    def _ensure_presence(self, assign: Dict[int, str], units: List[Any],
                         class_units: Dict[int, int]) -> List[str]:
        """保证每类在 val 和 test 里至少各有 1 个 unit，但要付出受控的代价。

        为什么需要：桶小时 7:2:1 会算出某集合为 0（3 个 unit 就是 2/1/0），
        稀有类因此在 test 里彻底消失，mAP 直接是 nan。

        为什么必须设预算：补位是拿 train 去填 test，一旦不设限，多个类各借一次
        就能把 train 掏空——4 个 unit 的场景里实测出现过 train 1 / val 1 / test 2，
        test 比 train 还大，整个划分失去意义。所以三条硬约束：

            * 总 unit 数 < 20 时**完全不补**：这么小的集合，补位只会破坏比例，
              宁可让报告如实显示某类在 test 缺失，由人去补数据；
            * 补位总量不超过总 unit 数的 5%（至少留 1 个额度）；
            * 每类只有在 unit 数 >= 5、且借完之后 train 里该类仍占 70% 以上时才借。

        返回逐条说明，供 balance_report 如实打印。
        """
        notes: List[str] = []
        total = len(units)
        if total < 20:
            notes.append("unit 只有 %d 个（<20），不做存在性补位："
                         "这个规模下补位会破坏 7:2:1，宁可如实报缺失" % total)
            return notes

        budget = max(1, int(round(total * 0.05)))
        moved = 0
        for c in sorted(class_units):
            holders = [u for u in units if c in u["classes"]]
            if len(holders) < 5:
                notes.append("类 %d 只有 %d 个 unit，不做强制覆盖"
                             % (c, len(holders)))
                continue
            for target in ("val", "test"):
                if moved >= budget:
                    notes.append("补位额度已用完（%d/%d），类 %d 在 %s 可能仍缺失"
                                 % (moved, budget, c, target))
                    break
                if any(assign[u["uid"]] == target for u in holders):
                    continue
                cand = sorted([u for u in holders
                               if assign[u["uid"]] == "train"],
                              key=lambda u: u["uid"])
                keep = max(1, int(round(0.7 * len(holders))))
                if len(cand) <= keep:
                    continue
                assign[cand[-1]["uid"]] = target
                moved += 1
                notes.append("类 %d 在原分配里没有 %s，从 train 借了 unit %d"
                             % (c, target, cand[-1]["uid"]))
        return notes

    def split(self, X: Any = None, Y: Any = None) -> Dict[str, List[Any]]:
        """执行分层抽样，返回 {'train': [...], 'val': [...], 'test': [...]}。

        参数不传就用上一次 split 的样本集；第一次调用必须显式传 X / Y
        （DataImporter.build_dataset 的两个返回值）。

        返回的下标可以直接拿去索引 X / Y。划分结果同时存进 self.splits，
        供 check_leakage / balance_report / write_splits 使用。

        算法：
            1) group_by_scene 把样本并成 unit（防三种泄漏）；
            2) 统计每类出现在多少个 unit 里；
            3) 每个 unit 按「最稀有类」归桶，桶内按 ratio 分（最大余数法）；
            4) 给稀有类补 val / test 的存在性；
            5) 自检泄漏，有问题直接抛。
        """
        import numpy as np

        X, Y = self._resolve(X, Y)

        units = self.group_by_scene(X, Y)
        class_units: Dict[int, int] = {}
        for u in units:
            for c in u["classes"]:
                class_units[c] = class_units.get(c, 0) + 1
        if not class_units:
            raise ValueError("所有样本都没有标注框，无法按类别分层")

        def bucket_of(u: Any) -> str:
            if not u["classes"]:
                return "空标注"
            rare = min(u["classes"], key=lambda c: (class_units[c], c))
            return "稀有类%02d" % rare

        buckets: Dict[str, List[Any]] = {}
        for u in units:
            buckets.setdefault(bucket_of(u), []).append(u)

        assign: Dict[int, str] = {}
        for key in sorted(buckets):
            items = sorted(buckets[key], key=lambda u: u["uid"])
            rng = np.random.default_rng([self.seed, zlib.crc32(key.encode())])
            order = rng.permutation(len(items))
            items = [items[int(i)] for i in order]
            n_tr, n_va, n_te = self._alloc(len(items))
            for k, u in enumerate(items):
                assign[u["uid"]] = ("train" if k < n_tr
                                    else "val" if k < n_tr + n_va else "test")

        self.notes = self._ensure_presence(assign, units, class_units)

        splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for u in units:
            splits[assign[u["uid"]]].extend(u["indices"])
        for k in splits:
            splits[k].sort()

        self.X, self.Y = X, Y
        self.assign = assign
        self.splits = splits

        report = self.check_leakage(X, Y)
        if not report["ok"]:
            raise ValueError("划分出现泄漏：" + "；".join(report["issues"]))
        return splits

    # ------------------------------------------------------------------
    # 3. 泄漏自检
    # ------------------------------------------------------------------
    def check_leakage(self, X: Any = None, Y: Any = None) -> Dict[str, Any]:
        """检查 group / 内容指纹 / 场景是否跨集合，返回自检报告。

        检查三样：
            unit   一个 unit 只允许出现在一个集合（这是结构保证，做冗余校验）；
            group  同一张原图的四个光照版不许分开（题面 §3 的核心要求）；
            content 内容完全相同的图不许分开（图库里有历史重复）；
            scene  同一场景批次不许分开（同背景同光照）。

        返回 {"ok": bool, "issues": [...], "counts": {...}}；不抛异常，
        由 split() 负责在看到 ok=False 时抛。想单独抽查时直接调它。
        """
        X, Y = self._resolve(X, Y)
        if not self.splits:
            raise ValueError("还没划分，先调用 split()")

        issues: List[str] = []
        unit_of = {}
        for u in self.units:
            for i in u["indices"]:
                unit_of[i] = u

        seen_unit: Dict[int, str] = {}
        group_where: Dict[Any, set] = {}
        scene_where: Dict[str, set] = {}
        content_where: Dict[str, set] = {}

        for name, idxs in self.splits.items():
            for i in idxs:
                u = unit_of.get(i)
                if u is None:
                    issues.append("样本 %d 不属于任何 unit" % i)
                    continue
                before = seen_unit.get(u["uid"])
                if before is not None and before != name:
                    issues.append("unit %d 同时出现在 %s 和 %s"
                                  % (u["uid"], before, name))
                seen_unit[u["uid"]] = name

                group_where.setdefault(X[i].get("group", i), set()).add(name)
                for sc in u["scenes"]:
                    scene_where.setdefault(sc, set()).add(name)
                for ck in u["contents"]:
                    content_where.setdefault(ck, set()).add(name)

        for g, where in group_where.items():
            if len(where) > 1:
                issues.append("group %s 跨集合：%s"
                              % (g, "/".join(sorted(where))))
        for sc, where in scene_where.items():
            if len(where) > 1:
                issues.append("场景 %s 跨集合：%s" % (sc, "/".join(sorted(where))))
        for ck, where in content_where.items():
            if len(where) > 1:
                issues.append("内容 %s… 跨集合：%s（重复图泄漏）"
                              % (ck[:8], "/".join(sorted(where))))

        counts = {k: {"samples": len(v),
                      "images": len({X[i].get("file") for i in v}),
                      "units": len({unit_of[i]["uid"] for i in v if i in unit_of})}
                  for k, v in self.splits.items()}
        return {"ok": not issues, "issues": issues, "counts": counts}

    # ------------------------------------------------------------------
    # 4. 类别分布报告
    # ------------------------------------------------------------------
    def balance_report(self, X: Any = None, Y: Any = None) -> Dict[str, Any]:
        """统计每类在三个集合里的图片数、框数、占比，并与 7:2:1 的偏差对比。

        题面 §3 要求「每类 >=120 张有效图、类别数量分布柱状图」，这份报告就是
        数据来源。偏差列用来判断分层是否真的起作用：健康的话每类在三个集合的
        占比都该贴着 ratio，偏差只在小类上稍微大一点。
        """
        X, Y = self._resolve(X, Y)
        if not self.splits:
            raise ValueError("还没划分，先调用 split()")

        names = DataImporter._load_class_names()
        all_cls = sorted({int(b[4]) for idxs in self.splits.values()
                          for i in idxs for b in Y[i]})

        rows: List[Dict[str, Any]] = []
        for c in all_cls:
            row: Dict[str, Any] = {"class_id": c,
                                   "name": names.get(c, str(c))}
            for part in ("train", "val", "test"):
                idxs = self.splits[part]
                n_img = sum(1 for i in idxs
                            if any(int(b[4]) == c for b in Y[i]))
                n_box = sum(1 for i in idxs for b in Y[i] if int(b[4]) == c)
                row[part + "_img"] = n_img
                row[part + "_box"] = n_box
            total = sum(row[p + "_box"] for p in ("train", "val", "test"))
            row["total_box"] = total
            row["share"] = [round(row[p + "_box"] / total, 3) if total else 0.0
                            for p in ("train", "val", "test")]
            row["dev"] = [round(row[p + "_box"] / total - self.ratio[k], 3)
                          if total else 0.0
                          for k, p in enumerate(("train", "val", "test"))]
            rows.append(row)

        print("划分：train %d / val %d / test %d 张样本，unit 共 %d 个"
              % (len(self.splits["train"]), len(self.splits["val"]),
                 len(self.splits["test"]), len(self.units)))
        print("按类别的图片数（train/val/test）：")
        for r in rows:
            print("  %-6s 类%d | %4d / %4d / %4d 张 | 框 %4d | 占比 %.3f/%.3f/%.3f"
                  " | 偏差 %+.3f/%+.3f/%+.3f"
                  % (r["name"], r["class_id"], r["train_img"], r["val_img"],
                     r["test_img"], r["total_box"], r["share"][0],
                     r["share"][1], r["share"][2], r["dev"][0], r["dev"][1],
                     r["dev"][2]))
        notes = list(getattr(self, "notes", []) or [])
        for n in notes:
            print("  [补位] " + n)

        total = sum(len(v) for v in self.splits.values())
        if total:
            actual = [len(self.splits[p]) / total
                      for p in ("train", "val", "test")]
            if max(abs(actual[i] - self.ratio[i]) for i in range(3)) > 0.08:
                warn = ("整体比例偏离 7:2:1 较多（实际 %.3f/%.3f/%.3f）："
                        "可分的 unit 太少，或场景划分太粗。补更多批次后重划。"
                        % tuple(actual))
                notes.append(warn)
                print("  [注意] " + warn)
        if not self.scene_map:
            warn = ("没提供 scene_map：同场景的不同原图会被当成不同场景拆到"
                    "不同集合，test mAP 会偏高。补一份 data/splits/scene_map.json"
                    " 再重跑。")
            notes.append(warn)
            print("  [注意] " + warn)

        return {"rows": rows, "ratio": list(self.ratio),
                "counts": {k: len(v) for k, v in self.splits.items()},
                "notes": notes}

    # ------------------------------------------------------------------
    # 5. 写清单
    # ------------------------------------------------------------------
    def write_splits(self, X: Any = None, out_dir: Optional[str] = None,
                     Y: Any = None) -> str:
        """把划分写成 data/splits/train.txt / val.txt / test.txt + splits.json。

        txt 每行一个文件名（同一张原图只写一次，不带 _dark 之类的后缀）；
        json 记 ratio、seed、每集合的样本/图片/unit 数、场景映射、类别分布，
        这些是报告「数据划分」一节的依据，也是复现的唯一凭据。
        """
        X, Y = self._resolve(X, Y)
        if not self.splits:
            raise ValueError("还没划分，先调用 split()")

        out = out_dir or self.splits_dir
        os.makedirs(out, exist_ok=True)

        files: Dict[str, List[str]] = {}
        for part, idxs in self.splits.items():
            names = sorted({str(X[i].get("file") or ("idx_%06d" % i))
                            for i in idxs})
            files[part] = names
            path = os.path.join(out, part + ".txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(n + "\n" for n in names))

        doc = {
            "ratio": list(self.ratio),
            "seed": self.seed,
            "size": X[0]["geom"]["size"] if X[0].get("geom") else None,
            "counts": {p: {"samples": len(v),
                           "images": len(files[p]),
                           "units": sum(1 for u in self.units
                                        if self.assign[u["uid"]] == p)}
                       for p, v in self.splits.items()},
            "units": [{"uid": u["uid"], "split": self.assign[u["uid"]],
                       "groups": u["groups"], "scenes": u["scenes"],
                       "classes": sorted(u["classes"]),
                       "n_samples": len(u["indices"])}
                      for u in self.units],
            "scene_map_used": bool(self.scene_map),
            "scene_of": {str(k): v for k, v in (self.scene_of or {}).items()},
            "files": files,
            "notes": list(getattr(self, "notes", []) or []),
        }
        with open(os.path.join(out, "splits.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        print("已写出：%s（train/val/test.txt + splits.json）" % out)
        return out

    # ------------------------------------------------------------------
    # 内部：取样本集
    # ------------------------------------------------------------------
    def _resolve(self, X: Any, Y: Any) -> Tuple[Any, Any]:
        """统一取样本集并做基本校验：没传的用上一次 split 的。"""
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if X is None or Y is None:
            raise ValueError(
                "没有样本集：第一次调用要显式传 X / Y"
                "（DataImporter.build_dataset 的两个返回值）")
        if len(X) != len(Y):
            raise ValueError("X 与 Y 长度不一致：%d vs %d" % (len(X), len(Y)))
        if len(X) == 0:
            raise ValueError("样本集为空，无法划分")
        return X, Y


# ---------------------------------------------------------------------------
# 3. 主干模型构建类
# ---------------------------------------------------------------------------
class ModelBuilder:
    """构建检测模型：主干 + 检测头，保证 CPU 可推理。

    题面 §4：可用现成架构（YOLOv5n / SSD-Lite / NanoDet 等）或自研小模型；
    用预训练权重必须写明来源与 license。§1#5、§7：现场无 GPU、实时 >=10 FPS。
    待定：框架、架构、输入分辨率、是否用预训练、类别数如何进检测头、
    参数量与 CPU 速度怎么权衡（要写进 docs/train.md）。
    """

    def __init__(self, config: Any = None, num_classes: int = 6,
                 input_size: int = 640, device: str = "cpu") -> None:
        self.config = None
        self.framework = "pytorch"          # 延迟导入，见 _import_torch()
        self.arch = "inception_v2 + 多级检测头"
        self.num_classes = None             # 6，由 classes.txt 决定
        self.input_size = None
        self.pretrained = None              # 权重路径 + 来源 + license
        self.device = None                  # 训练设备；推理端固定 CPU
        self.backbone = None
        self.head = None
        self.model = None
        self.out_stages = None              # 取出特征的那几级下标，默认 (2,3,4)
        self.out_channels = None            # 这几级输出的通道 (192, 256, 320)
        self.out_strides = None             # 这几级的 stride (8, 16, 32)
        self.feat_hw = None                 # 这几级的特征图尺寸 [(80,80),(40,40),(20,20)]
        self.head_out_channels = None       # 检测头每层输出通道 4 + num_classes(+1)
        self.has_objectness = None          # 检测头是否带独立 objectness 通道
        self.stage_channels = None          # 每一级的输出通道（主干全部级）
        self.stem_ch = None                 # stem 输出通道
        self._torch = None                  # 延迟导入的 torch 模块（缓存）
        self._nn = None                     # 延迟导入的 torch.nn（缓存）

        # ---- 参数校验：这一类的错误要在这里就炸，不要等到 forward ----
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes 要是正整数：%r" % (num_classes,))
        if not isinstance(input_size, int) or input_size < 32:
            raise ValueError("input_size 要是 >=32 的整数：%r" % (input_size,))
        if input_size % 32 != 0:
            raise ValueError(
                "input_size 必须能被最大 stride 32 整除，否则三级特征尺寸对不上：%d"
                % input_size)
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 只支持 'auto' / 'cpu' / 'cuda'：%r" % (device,))

        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.device = device
        self.config = config

    # ---- 3.1 卷积层：Inception-V2 四支路模块 --------------------------------
    def _import_torch(self) -> Tuple[Any, Any]:
        """延迟导入 PyTorch：没装时只在使用建模功能时才报错，import train 不受影响。"""
        if self._torch is not None:
            return self._torch, self._nn
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "没装 PyTorch，建不了图。现场无 GPU，装 CPU 版就够：\n"
                "  python3 -m pip install --no-cache-dir torch "
                "--index-url https://download.pytorch.org/whl/cpu") from exc
        self._torch, self._nn = torch, nn
        return torch, nn

    def inception_v2_block(self, in_ch: int, out_ch: int, stride: int = 1,
                           factorize: bool = True, reduce_ratio: int = 4) -> Any:
        """搭一个 Inception-V2 的四支路卷积模块，每层都是 conv -> BN -> ReLU。

        出处：Szegedy 等《Rethinking the Inception Architecture》（Inception-v2/v3）。
        四条支路并行跑，最后在通道维拼起来，所以同一层同时看四种邻域：

            支路1   1x1 卷积                          跨通道组合，不看邻域
            支路2   1x1 降维 -> 3x3                    小邻域
            支路3   1x1 降维 -> 3x3 -> 3x3 或 1x3+3x1   大邻域（5x5 用两个 3x3 代替）
            支路4   3x3 平均池化 -> 1x1                先不学参数地做平滑，再跨通道组合

        为什么选它（写报告时直接用，别照抄，按你实测的数改）:
            * 多支路 = 多感受野并行。你的 6 类目标 sqrt(w*h) 跨约 5 倍，
              单一大核或多级下采样都要靠堆层数去覆盖，四支路一层就能同时拿到
              1x1 / 3x3 / 5x5 等效邻域，对书（大平面）和笔（细长条）同时友好。
            * 5x5 用两个 3x3 代替：参数从 25 降到 18（每通道对），
              再拆成 1x3 + 3x1 只剩 6，算力换得更彻底，适合 CPU 端。
            * 每个卷积后面都跟 BN：你只有三百多张原始图，batch 小、分布被
              三种光照增强拉开了，没有 BN 很难收敛；BN 让每层输入保持零均值
              单位方差，学习率可以放大，也降低了对初始化的敏感度。
            * bias=False：卷积后面紧跟 BN，BN 会减掉均值再加 beta，
              卷积的 bias 被完全吸收，留着纯属多余参数。

        代价（写进 docs/train.md 的取舍部分）:
            同为 out_ch 输出，四支路的参数量和算力大约是单个 3x3 的 2~3 倍。
            所以它适合放在分辨率已经降下来的位置（stride >= 8）；
            放在 stride 4 的高分辨率上会明显拖慢 CPU 推理。

        参数:
            in_ch        输入通道数
            out_ch       输出通道数 = 四支路拼接后的通道数，必须 >= 4
            stride       1 或 2。2 时四条支路一起降一半分辨率（v2 的降采样用法）
            factorize    True：支路3 的第二个 3x3 拆成 1x3 + 3x1（v2 的 n×n 分解）
                         False：保留 3x3（更接近原始 Inception 模块）
            reduce_ratio 支路2/3 内部 1x1 的降维比例，默认 4，即降到 in_ch/4

        返回:
            nn.Module，输入 [B, in_ch, H, W] -> 输出 [B, out_ch, H/stride, W/stride]
        """
        # ---- 参数校验：先校验再导入 torch，传错参数不该先报「没装 torch」 ----
        if not isinstance(in_ch, int) or in_ch < 1:
            raise ValueError("in_ch 要是正整数：%r" % (in_ch,))
        if not isinstance(out_ch, int) or out_ch < 4:
            raise ValueError("out_ch 至少 4（四条支路各占一份）：%r" % (out_ch,))
        if stride not in (1, 2):
            raise ValueError("stride 只支持 1 或 2：%r" % (stride,))
        if not isinstance(reduce_ratio, int) or reduce_ratio < 1:
            raise ValueError("reduce_ratio 要是正整数：%r" % (reduce_ratio,))

        torch, nn = self._import_torch()

        base = out_ch // 4
        split = [base, base, base, out_ch - 3 * base]   # 余数给支路4（最便宜）
        mid = max(1, in_ch // reduce_ratio)             # 支路2/3 的内部窄通道数

        class ConvBNReLU(nn.Module):
            """conv -> BN -> ReLU。bias=False，偏置由 BN 的 beta 承担。"""

            def __init__(self, ci: int, co: int, k: Any, s: int = 1) -> None:
                super().__init__()
                if isinstance(k, tuple):
                    pad = tuple(v // 2 for v in k)      # (1,3)/(3,1) 要逐维算填充
                else:
                    pad = k // 2
                self.conv = nn.Conv2d(ci, co, k, s, pad, bias=False)
                self.bn = nn.BatchNorm2d(co)
                self.act = nn.ReLU(inplace=True)

            def forward(self, x: Any) -> Any:
                return self.act(self.bn(self.conv(x)))

        class InceptionV2(nn.Module):
            """四支路并行 + 通道拼接。"""

            def __init__(self) -> None:
                super().__init__()
                c1, c2, c3, c4 = split

                # 支路1：1x1，只看跨通道组合
                self.b1 = ConvBNReLU(in_ch, c1, 1, stride)

                # 支路2：1x1 降维 -> 3x3，看小邻域
                self.b2 = nn.Sequential(
                    ConvBNReLU(in_ch, mid, 1),
                    ConvBNReLU(mid, c2, 3, stride),
                )

                # 支路3：1x1 降维 -> 3x3 -> （1x3 + 3x1 或 3x3），看大邻域
                if factorize:
                    tail = nn.Sequential(
                        ConvBNReLU(mid, mid, (1, 3)),
                        ConvBNReLU(mid, c3, (3, 1), stride),
                    )
                else:
                    tail = nn.Sequential(
                        ConvBNReLU(mid, mid, 3),
                        ConvBNReLU(mid, c3, 3, stride),
                    )
                self.b3 = nn.Sequential(ConvBNReLU(in_ch, mid, 1), tail)

                # 支路4：3x3 平均池化 -> 1x1，不学参数地先平滑
                # count_include_pad=False：边界不用 0 填，避免灰边被当成暗色特征
                self.b4 = nn.Sequential(
                    nn.AvgPool2d(3, stride, 1, count_include_pad=False),
                    ConvBNReLU(in_ch, c4, 1),
                )

            def forward(self, x: Any) -> Any:
                return torch.cat([self.b1(x), self.b2(x),
                                  self.b3(x), self.b4(x)], dim=1)

        return InceptionV2()

    # ---- 3.2 池化层：2x2 最大池化 ------------------------------------------
    def max_pool_block(self, kernel_size: int = 2, stride: int = 2) -> Any:
        """2x2 最大池化（stride 2、无填充），分辨率正好减半。

        为什么用最大值、为什么只用 2x2：
            * 取窗口内的最大值 = 只留最强的那个响应。检测问的是「这里有没有
              书脊 / 把手」，弱响应多半是背景纹理；平均池化会把它摊平进特征，
              正好抹掉书↔草稿本、杯子↔保温杯靠局部结构区分的那点差别。
            * 2x2 stride 2 是不重叠的整数下采样：H、W 严格变成 H/2、W/2，
              不像 3x3 池化那样在边界上留 1 像素的取整歧义，多级特征图的
              尺寸可以直接心算，接检测头时不用反复对齐。
            * 不填充、不 ceil：input_size 已在上面的 __init__ 里校验成 32 的倍数，
              一路除 2 到底都是整数，最后一层不会多出半格。
            * 它同时兼两件事：降分辨率省算力（CPU 端必须），以及给尺度分带
              ——相邻两级特征差 2 倍，这是不建图像金字塔也能覆盖 5 倍尺度差的依据。

        参数:
            kernel_size  池化窗口边长，默认 2
            stride       步长，默认 2（与窗口相等即不重叠）
        返回:
            nn.MaxPool2d，输入 [B,C,H,W] -> 输出 [B,C,H//2,W//2]
        """
        if not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError("kernel_size 要是正整数：%r" % (kernel_size,))
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride 要是正整数：%r" % (stride,))

        _, nn = self._import_torch()
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                            padding=0, ceil_mode=False)

    # ---- 3.3 全连接层：拉平 + 两层 FC ---------------------------------------
    def fc_block(self, in_features: int, hidden_dim: int,
                 out_features: int) -> Any:
        """把特征图拉成一维，接两层全连接：flatten -> Linear -> ReLU -> Linear。

        为什么这样接（按你的要求：拉平 + 两层）：
            * flatten 从 dim=1 开始压，保留 batch 维： [B,C,H,W] -> [B,C*H*W]。
            * 第一层 Linear 做的是「全局线性组合」——它把每个空间位置、每个
              通道的值都当成一个独立输入，所以能利用整幅图的全局信息，
              这一点是卷积（局部核）本身给不了的。
            * 中间夹 ReLU：两层 Linear 直接连，等效于一个线性映射
              （W2(W1 x + b1) + b2 = W x + b，秩被 min(hidden) 卡住），
              等于白搭一层；ReLU 才让它有非线性。
            * 第二层不再接激活：它是输出层。分类要 logits（交给后面的
              softmax / 交叉熵），回归要能出负数，都不该被 ReLU 截断。

        参数:
            in_features   flatten 后的长度，必须等于 C*H*W
                          （例：512 通道 × 20 × 20 = 204800）
            hidden_dim    中间层宽度
            out_features  输出维数（例：6 类 × 每格 5 个数 = 30）

        返回:
            nn.Module，输入 [B,C,H,W] -> 输出 [B, out_features]

        注意（不是现在要改，先记着，写 docs/train.md 时用得上）:
            in_features 一旦定死，输入分辨率就被绑定了，换 input_size 必须
            重建这层；而且参数量 = in_features × hidden_dim，20×20 的特征图
            会直接上千万参数，CPU 端推理会明显变慢。
        """
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError("in_features 要是正整数：%r" % (in_features,))
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim 要是正整数：%r" % (hidden_dim,))
        if not isinstance(out_features, int) or out_features < 1:
            raise ValueError("out_features 要是正整数：%r" % (out_features,))

        torch, nn = self._import_torch()

        class FCHead(nn.Module):
            """flatten -> Linear -> ReLU -> Linear。"""

            def __init__(self) -> None:
                super().__init__()
                self.flatten = nn.Flatten(1)        # 只压 C,H,W，batch 留着
                self.fc1 = nn.Linear(in_features, hidden_dim)
                self.act = nn.ReLU(inplace=True)
                self.fc2 = nn.Linear(hidden_dim, out_features)
                # 两层都保留 bias：FC 后面没有 BN 帮忙吸收，偏置必须留着

            def forward(self, x: Any) -> Any:
                x = self.flatten(x)
                return self.fc2(self.act(self.fc1(x)))

        return FCHead()

    # ---- 3.3 残差块：把两个 Inception 串起来，加一条恒等捷径 ---------------
    def residual_block(self, in_ch: int, out_ch: int, stride: int = 1,
                       factorize: bool = True,
                       reduce_ratio: int = 4) -> Any:
        """Inception 块 + 恒等捷径：y = ReLU(x + F(x))。

        什么时候才该加残差：只有「同一分辨率上连续堆两个以上块」才有意义。
        单个块加残差，x 和 F(x) 形状都不一样，还得补一条 1x1 投影，纯亏参数。
        残差真正解决的是深度的副作用 —— 反向传播时

            ∂L/∂x = ∂L/∂y · (I + ∂F/∂x)

        那个单位矩阵保证梯度永远有一条原样回传的路，不会因为 ∂F/∂x 太小
        而在几十层连乘后消失。

        通道相同（同级的第二个块）时直接相加，不花任何参数；
        通道不同（级的第一个块）才用 1x1 卷积投影对齐。
        """
        torch, nn = self._import_torch()
        inner = self.inception_v2_block(in_ch, out_ch, stride=stride,
                                        factorize=factorize,
                                        reduce_ratio=reduce_ratio)
        if in_ch == out_ch and stride == 1:
            shortcut = nn.Identity()
        else:
            # 投影捷径：把通道数和步长对齐，否则加不起来
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        class Residual(nn.Module):
            """y = ReLU(BN(捷径(x) + 主干(x)))。

            相加之后那个批量归一化不能省：两条支路的输出各自被 BN 归一化过，
            但加起来幅度会累加，十几个块串下来特征就整体放大几十倍，检测头
            的 logit 跟着爆炸 —— 表现就是训练第一轮置信度损失飙到 30 以上
            （正常应该在 4~8）。加一次 BN 把幅度拉回来，梯度也稳。
            """

            def __init__(self) -> None:
                super().__init__()
                self.main = inner
                self.skip = shortcut
                self.bn = nn.BatchNorm2d(out_ch)
                self.act = nn.ReLU(inplace=True)

            def forward(self, x: Any) -> Any:
                return self.act(self.bn(self.skip(x) + self.main(x)))

        return Residual()

    # ---- 3.4 主干：卷积堆叠 + 池化 + 全连接 --------------------------------
    def build_backbone(self, channels: Any = (64, 128, 192, 256, 320, 384),
                       stem_ch: int = 32,
                       out_stages: Any = (2, 3, 4),
                       blocks_per_stage: Any = (1, 1, 2, 2, 3, 3),
                       residual: bool = True) -> Any:
        """搭主干，在指定的几级引出特征交给检测头（不再接全连接）。

        结构（input_size=640、channels 默认 6 级时的实际尺寸）:

            层        内容                      输出尺寸    通道   取样
            --------------------------------------------------------------
            stem      3x3 conv s2 + BN + ReLU   320x320      32
            block0    Inception-V2 (stride 1)    320x320      64
            down0     2x2 maxpool s2             160x160      64
            block1    Inception-V2               160x160     128
            down1     2x2 maxpool s2              80x80     128
            block2    Inception-V2                80x80     192   -> P3  s=8
            down2     2x2 maxpool s2              40x40     192
            block3    Inception-V2                40x40     256   -> P4  s=16
            down3     2x2 maxpool s2              20x20     256
            block4    Inception-V2                20x20     320   -> P5  s=32
            down4     2x2 maxpool s2              10x10     320
            block5    Inception-V2                10x10     384

        三个刻意的选择:

            * 取样点在 block 之后、down 之前。block 的输出才是这一级真正的
              特征（通道 192/256/320）；如果在 down 之后再取，80x80 会变成
              40x40（stride 16），P3 的 8 倍下采样就对不上了。
            * block 和它后面的 down 分开存成两个 ModuleList，不塞进一个
              nn.Sequential。Sequential 一跑到底，中途拿不到任何中间张量；
              拆开才能在第 i 级插入取样点。
            * 取样级用下标 out_stages 指定，不是"取最后三级"写死。默认
              (2,3,4) 对应 stride 8/16/32；以后想换成 16/32/64 只需传 (3,4,5)，
              通道和 stride 会自动跟着算，函数体不用改。
            * 最后一级后面的 down 其实不产生任何输出（往下没人接），保留它是
              为了 blocks 和 downs 一一对应，增删级数时不用写特判。它只是
              10x10 -> 5x5 的池化，代价可以忽略。

        参数:
            channels    每一级的输出通道，元组，长度 = Inception 级数
            stem_ch     stem 输出通道
            out_stages  取样级下标（0 基），默认 (2,3,4) = stride 8/16/32
        返回:
            nn.Module，输入 [B,3,input_size,input_size] -> 元组 (p3, p4, p5)，
            每个是 [B,C,H,W] 的中间特征，还没解码、不含任何分类/回归分支
        """
        chans = tuple(channels) if isinstance(channels, (tuple, list)) else ()
        if not chans:
            raise ValueError("channels 要是非空的通道数序列：%r" % (channels,))
        for c in chans:
            if not isinstance(c, int) or c < 4:
                raise ValueError("channels 里每个数都要 >=4（四支路各占一份）：%r"
                                 % (c,))
        if not isinstance(stem_ch, int) or stem_ch < 1:
            raise ValueError("stem_ch 要是正整数：%r" % (stem_ch,))

        stages = tuple(out_stages) if isinstance(out_stages, (tuple, list)) \
            else (out_stages,)
        if not stages:
            raise ValueError("out_stages 不能为空：%r" % (out_stages,))
        for i in stages:
            if isinstance(i, bool) or not isinstance(i, int):
                raise ValueError("out_stages 里要放整数下标：%r" % (i,))
            if not 0 <= i < len(chans):
                raise ValueError("out_stages 下标越界，要在 0~%d 之间：%r"
                                 % (len(chans) - 1, i))
        if len(set(stages)) != len(stages):
            raise ValueError("out_stages 不能有重复下标：%r" % (out_stages,))

        torch, nn = self._import_torch()

        # ---- stem：3 通道抬到 stem_ch ----
        stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # ---- 每一级的块序列与池化分开注册，才好在级与级之间取样 ----
        # 第 i 级的第 0 个块负责把通道从 prev 改成 c（形状变了，不能直接相加），
        # 之后的块通道不变，套上残差捷径。
        depth = tuple(blocks_per_stage) if isinstance(blocks_per_stage, (tuple, list)) \
            else tuple([blocks_per_stage] * len(chans))
        if len(depth) != len(chans):
            raise ValueError("blocks_per_stage 有 %d 项，channels 有 %d 项，要一一对应"
                             % (len(depth), len(chans)))
        for d in depth:
            if isinstance(d, bool) or not isinstance(d, int) or d < 1:
                raise ValueError("blocks_per_stage 里每项要是 >=1 的整数：%r" % (d,))

        stage_mods, downs = [], []
        prev = stem_ch
        for c, n in zip(chans, depth):
            seq = []
            for j in range(n):
                if j == 0:
                    # 改通道 -> 不加残差（补投影纯亏参数）
                    seq.append(self.inception_v2_block(prev, c, stride=1))
                else:
                    # 同通道 -> 直接相加，零额外参数的恒等路径
                    seq.append(self.residual_block(c, c) if residual
                               else self.inception_v2_block(c, c))
            stage_mods.append(nn.Sequential(*seq))
            downs.append(self.max_pool_block(2, 2))
            prev = c

        # 下标 i 那一级的 stride：stem 先降 2 倍，之后每过一级再乘 2
        stride_of = tuple(2 ** (i + 1) for i in stages)

        class Backbone(nn.Module):
            """stem -> 每级 [若干块] -> pool，按 out_stages 吐出中间特征。"""

            def __init__(self) -> None:
                super().__init__()
                self.stem = stem
                self.stages = nn.ModuleList(stage_mods)
                self.downs = nn.ModuleList(downs)
                self.out_stages = stages

            def forward(self, x: Any) -> Any:
                x = self.stem(x)
                outs = []
                for i, seq in enumerate(self.stages):
                    x = seq(x)                  # 这一级的全部块跑完
                    if i in self.out_stages:    # 取这一级最终的特征
                        outs.append(x)
                    x = self.downs[i](x)        # 降到下一级
                return tuple(outs)

        model = Backbone()

        # ---- 假图跑一遍量出每级实际尺寸：改 input_size / channels 都不用改代码 ----
        # 必须 eval：最后一层只有 5x5，train 模式下 BN 拿它做批统计，batch=1
        # 时统计量等于单样本自身，行为不可靠
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                probe = model(torch.zeros(1, 3, self.input_size, self.input_size))
            feat_hw = tuple(tuple(int(v) for v in p.shape[2:]) for p in probe)
            out_channels = tuple(int(p.shape[1]) for p in probe)
            del probe
        finally:
            model.train(was_training)

        # ---- 记录实例状态，供 build_head / summary / 文档引用 ----
        self.backbone = model
        self.out_stages = stages
        self.out_channels = out_channels
        self.out_strides = stride_of
        self.feat_hw = feat_hw
        self.stage_channels = chans
        self.stem_ch = stem_ch
        self.arch = ("inception_v2 每级%s块%s (%s) -> P%s @ stride %s, ch %s"
                     % ("/".join(str(d) for d in depth),
                        "含残差" if residual else "无残差",
                        "/".join(str(c) for c in chans),
                        "/".join(str(i) for i in stages),
                        "/".join(str(s) for s in stride_of),
                        "/".join(str(c) for c in out_channels)))
        return model

    # ---- 3.5 检测头：三级网格 + anchor-free 解码 ----------------------------
    def build_head(self, in_channels: Any = None, feat_hw: Any = None,
                   strides: Any = None, hidden_ch: int = 64,
                   with_objectness: bool = True,
                   p3_kernel: int = 3) -> Any:
        """三级检测头：每层出 [B, 4+num_classes(+1), H, W]，并附一个 decode()。

        每个格点摊开的通道顺序（6 类时共 11 个）:

            0 : 4                框的原始回归量 (tx, ty, tw, th)，未解码
            4 : 4+num_classes    每个类别的 logit，未过 sigmoid
            最后 1 个（可选）     objectness logit：这一格有没有目标

        objectness 放最后，是为了让类别切片 4:4+nc 在带不带它两种情况下完全
        一样，损失函数不用分情况。想去掉就传 with_objectness=False，通道变 10，
        其余代码一行都不用改。

        为什么 anchor-free：预设 anchor 需要先聚类出宽高，而你的 6 类目标尺寸
        跨约 5 倍（细长的笔 vs 大平面的书），anchor 要么数量爆炸要么覆盖不全。
        直接让每个格点回归"框的中心偏移 + 尺寸"就没有这个问题，代价是每个
        格点都要学会判断"我该不该负责这个目标"。

        四个刻意的选择:

            * forward 返回未解码的 raw，解码单独放在 decode()。因为分类损失
              必须在 logits 上算（BCEWithLogits），forward 如果直接吐概率，
              损失就没得用；反过来若给损失和推理各写一份解码，两边公式稍不
              一致，训完框会整体偏且极难查。所以约定：forward=raw，decode 是
              唯一的解码出口，公式只写一份。
            * 网格坐标注册成 buffer 且 persistent=False。这样 model.to(device)
              会带着它一起搬，导出 ONNX 时会被内联成常量，也不会跑进
              state_dict 被当成需要训练的权重。
            * P3 的 stem 是「1x1 降维 + 3x3」，不是直接 3x3。192 通道直接做
              3x3 要 0.707 GMACs，先降到 64 再做 3x3 只要 0.315 GMACs，
              省掉一半多，而分辨率高的这一层正是全场最贵的地方。
            * 分类分支的 bias 按「正样本先验 1%」初始化（-log(99) ≈ -4.595）。
              不这么做的话，早期 8400 个格点的分类输出都在 0 附近，sigmoid 后
              全接近 0.5，一半格点一上来就被当成正样本，梯度会被背景淹没。

        参数:
            in_channels     三层输入通道；None 时用 build_backbone 记录的
            feat_hw         三层特征图尺寸 [(H,W),...]；None 时优先用记录的，
                            没记录就按 input_size // stride 推算
            strides         三层 stride；None 时用 build_backbone 记录的
            hidden_ch       头内中间通道
            with_objectness 是否带独立 objectness 通道
            p3_kernel       最细那层 stem 的卷积核，3 或 1（1 表示省掉那层 3x3）
        返回:
            nn.Module。两种调用方式:
                head(feats)   feats=(p3,p4,p5) -> 元组，每层 [B, C, H, W]
                head.decode(raws) -> (boxes, scores, objectness, levels)
                    boxes       [B, N, 4]  xyxy，**letterbox 后的输入尺度**（0~sz）
                    scores      [B, N, nc] 已过 sigmoid 的类别分
                    objectness  [B, N, 1] 或 None
                    levels      [B, N] 每个框来自哪一级（0/1/2）
                    N = 三层格点数之和（640 输入时 6400+1600+400 = 8400）
            还原到原图尺寸需要 letterbox 的 scale / pad，那是 OutputExporter 的事。
        """
        # ---- 参数校验（先校验，再导入 torch） ----
        if in_channels is None:
            src = self.out_channels
            in_ch = tuple(src) if src is not None else (192, 256, 320)
        else:
            in_ch = tuple(in_channels) if isinstance(in_channels, (tuple, list)) \
                else (in_channels,)
        if strides is None:
            st = tuple(self.out_strides) if self.out_strides is not None \
                else (8, 16, 32)
        else:
            st = tuple(strides) if isinstance(strides, (tuple, list)) \
                else (strides,)
        if not in_ch or not isinstance(in_ch[0], int):
            raise ValueError("in_channels 要是非空的整数序列：%r" % (in_channels,))
        for c in in_ch:
            if isinstance(c, bool) or not isinstance(c, int) or c < 1:
                raise ValueError("in_channels 里每个数要是正整数：%r" % (c,))
        for s in st:
            if isinstance(s, bool) or not isinstance(s, int) or s < 1:
                raise ValueError("strides 里每个数要是正整数：%r" % (s,))
        if len(st) != len(in_ch):
            raise ValueError("strides 有 %d 个，in_channels 有 %d 个，要一一对应"
                             % (len(st), len(in_ch)))
        if not isinstance(hidden_ch, int) or hidden_ch < 1:
            raise ValueError("hidden_ch 要是正整数：%r" % (hidden_ch,))
        if p3_kernel not in (1, 3):
            raise ValueError("p3_kernel 只支持 1 或 3：%r" % (p3_kernel,))

        if feat_hw is None:
            hw = tuple(self.feat_hw) if self.feat_hw is not None else \
                tuple((self.input_size // s, self.input_size // s) for s in st)
        else:
            hw = tuple(tuple(int(v) for v in one) for one in feat_hw)
        if len(hw) != len(in_ch):
            raise ValueError("feat_hw 有 %d 组，in_channels 有 %d 个，要一一对应"
                             % (len(hw), len(in_ch)))
        for one in hw:
            if len(one) != 2 or one[0] < 1 or one[1] < 1:
                raise ValueError("feat_hw 每组要是 (H, W) 两个正整数：%r" % (one,))

        torch, nn = self._import_torch()

        nc = int(self.num_classes)
        n_obj = 1 if with_objectness else 0
        out_ch = 4 + nc + n_obj
        use_p3_conv = p3_kernel == 3

        class DetectHead(nn.Module):
            """三级 anchor-free 检测头。forward 出 raw，decode 出像素框。"""

            def __init__(self) -> None:
                super().__init__()
                self.num_classes = nc
                self.with_objectness = with_objectness
                self.strides = tuple(st)
                self.feat_hw = tuple(hw)
                self.out_channels = out_ch

                stems, regs, clss = [], [], []
                for k, cin in enumerate(in_ch):
                    # 只给最细的一层加 3x3：它负责小目标，需要看邻域才能把
                    # 细长目标的两端连起来；P4/P5 上的目标已经够大，3x3 不值那个算力
                    layers = [
                        nn.Conv2d(cin, hidden_ch, 1, bias=False),
                        nn.BatchNorm2d(hidden_ch),
                        nn.SiLU(inplace=True),
                    ]
                    if k == 0 and use_p3_conv:
                        layers += [
                            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(hidden_ch),
                            nn.SiLU(inplace=True),
                        ]
                    stems.append(nn.Sequential(*layers))
                    regs.append(nn.Conv2d(hidden_ch, 4, 1, bias=True))
                    clss.append(nn.Conv2d(hidden_ch, nc + n_obj, 1, bias=True))

                self.stems = nn.ModuleList(stems)
                self.regs = nn.ModuleList(regs)
                self.clss = nn.ModuleList(clss)

                # 网格：格点中心以"格"为单位的整数下标 (j, i)，形状 [1,2,H,W]。
                # 解码时用它把 tx/ty 的偏移量落到正确的格子上。
                for k, (h, w) in enumerate(hw):
                    gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w),
                                            indexing="ij")
                    grid = torch.stack([gx, gy], dim=0).float().unsqueeze(0)
                    self.register_buffer("grid_%d" % k, grid, persistent=False)

                # 分类分支 bias：正样本先验 1%，sigmoid(-4.595) ≈ 0.01
                prior = 0.01
                cls_bias = -float(torch.log(torch.tensor((1.0 - prior) / prior)))
                for m in self.clss:
                    nn.init.constant_(m.bias, cls_bias)
                for m in self.regs:
                    nn.init.constant_(m.bias, 0.0)

            def forward(self, feats: Any) -> Any:
                if len(feats) != len(self.stems):
                    raise ValueError("特征有 %d 层，检测头是按 %d 层建的"
                                     % (len(feats), len(self.stems)))
                raws = []
                for k, f in enumerate(feats):
                    x = self.stems[k](f)
                    raws.append(torch.cat([self.regs[k](x), self.clss[k](x)],
                                          dim=1))
                return tuple(raws)

            def decode(self, raws: Any) -> Any:
                """把 raw 解码成输入尺度下的框。不做阈值过滤，也不做 NMS。"""
                boxes_all, score_all, obj_all, lvl_all = [], [], [], []
                for k, raw in enumerate(raws):
                    b, _, h, w = raw.shape
                    s = float(self.strides[k])
                    grid = getattr(self, "grid_%d" % k).to(raw.dtype)
                    t = raw[:, 0:4]

                    # 中心：格点只会认领自己格内（含半格余量）的中心，
                    # sigmoid 把偏移量压在 (0,1)，乘 2 减 0.5 后落在 (-0.5, 1.5) 格
                    cx = (grid[:, 0] + 2.0 * torch.sigmoid(t[:, 0]) - 0.5) * s
                    cy = (grid[:, 1] + 2.0 * torch.sigmoid(t[:, 1]) - 0.5) * s

                    # 尺寸：log 空间回归，exp 保证为正；必须钳位，否则早期几步
                    # tw 一冲高就 exp 成 inf。上界取 5：输入 640、s=8 时 w/s 的
                    # 理论上限是 80，log(80)≈4.4，5 够用
                    bw = torch.exp(torch.clamp(t[:, 2], -5.0, 5.0)) * s
                    bh = torch.exp(torch.clamp(t[:, 3], -5.0, 5.0)) * s

                    box = torch.stack([cx - bw * 0.5, cy - bh * 0.5,
                                       cx + bw * 0.5, cy + bh * 0.5], dim=-1)
                    boxes_all.append(box.reshape(b, -1, 4))

                    cls = raw[:, 4:4 + self.num_classes]
                    cls = cls.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
                    score_all.append(torch.sigmoid(cls))

                    if self.with_objectness:
                        ob = raw[:, 4 + self.num_classes:]
                        ob = ob.permute(0, 2, 3, 1).reshape(b, -1, 1)
                        obj_all.append(torch.sigmoid(ob))

                    lvl_all.append(torch.full((b, h * w), k, dtype=torch.long,
                                              device=raw.device))

                boxes = torch.cat(boxes_all, dim=1)
                scores = torch.cat(score_all, dim=1)
                objs = torch.cat(obj_all, dim=1) if obj_all else None
                levels = torch.cat(lvl_all, dim=1)
                return boxes, scores, objs, levels

        model = DetectHead()

        self.head = model
        self.head_out_channels = out_ch
        self.has_objectness = bool(with_objectness)
        return model

    def load_pretrained(self) -> Any:
        """加载预训练权重（可选）。没有就明确说「从头训练」，不留悬念。

        这个模型是自研的小结构，公开数据集上没有对得上的预训练权重，所以默认
        self.pretrained = None，即从头训练。题面 §4 只要求「用预训练权重必须
        写明来源与 license」，不用就不需要写。

        如果以后要接着自己之前的权重训（迁移/微调），把 .pt 路径塞进
        self.pretrained，这里会按 backbone / head 分别加载。
        """
        torch, _ = self._import_torch()
        if not self.pretrained:
            self.arch += "（从头训练，未用预训练权重）"
            return None
        if self.backbone is None or self.head is None:
            raise RuntimeError("先 build_backbone / build_head，再加载权重")
        blob = torch.load(self.pretrained, map_location="cpu")
        if isinstance(blob, dict) and "backbone" in blob and "head" in blob:
            self.backbone.load_state_dict(blob["backbone"])
            self.head.load_state_dict(blob["head"])
        else:
            self.backbone.load_state_dict(blob)
        return {"path": self.pretrained, "loaded": True}

    def assemble(self) -> Any:
        """把主干和检测头拼成一个模块：输入图，输出三层 raw。

        为什么要单独拼一步：训练循环、导出、检测三处都需要「图进、raw 出」
        这一个动作。拼成模块之后，torch.onnx.export 才有东西可导，训练循环
        也才能直接用 model(x)。
        """
        torch, nn = self._import_torch()
        if self.backbone is None:
            self.build_backbone()
        if self.head is None:
            self.build_head()
        backbone, head = self.backbone, self.head

        class DetectModel(nn.Module):
            """backbone + head。forward 只出 raw，解码仍由 head.decode 负责。"""

            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x: Any) -> Any:
                return self.head(self.backbone(x))

        self.model = DetectModel()
        self.model.train()
        return self.model

    def to_device(self) -> Any:
        """把模型放到 self.device；返回模型本身，方便链式调用。

        训练在显卡上做，导出前要能干净地转回 CPU（题面 §6：现场无 GPU）。
        """
        if self.model is None:
            self.assemble()
        dev = self.device
        if dev == "auto":
            torch, _ = self._import_torch()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = dev
        self.model.to(torch.device(dev))
        return self.model

    def count_params(self) -> Any:
        """统计参数量，回答「为什么选这个模型」。"""
        if self.model is None and self.backbone is None:
            raise RuntimeError("还没建模型")
        def _n(mod):
            return sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        total = _n(self.model) if self.model is not None else (
            _n(self.backbone) + _n(self.head))
        trainable = 0
        mods = [self.model] if self.model is not None else [self.backbone, self.head]
        for m in mods:
            if m is not None:
                trainable += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable),
                "backbone": int(_n(self.backbone)), "head": int(_n(self.head)),
                "million": round(total / 1e6, 3)}

    def forward_shape_check(self) -> Any:
        """用一张假图跑一遍前向，确认形状与输出格式对得上。

        这一步是训练开始前的最后一道关：形状不对要在这里炸，不要等训到
        第 3 轮才发现损失算不了。
        """
        torch, _ = self._import_torch()
        if self.model is None:
            self.assemble()
        was = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                x = torch.zeros(1, 3, self.input_size, self.input_size)
                x = x.to(next(self.model.parameters()).device)
                raws = self.model(x)
        finally:
            self.model.train(was)
        shapes = [tuple(int(v) for v in r.shape) for r in raws]
        expect = 4 + self.num_classes + (1 if self.has_objectness else 0)
        for s in shapes:
            if len(s) != 4 or s[1] != expect:
                raise ValueError("输出形状 %r 的通道数与 4+类别数(+1)=%d 不符"
                                 % (s, expect))
        return {"input": (1, 3, self.input_size, self.input_size),
                "outputs": shapes,
                "grid_points": sum(s[2] * s[3] for s in shapes)}

    def summary(self) -> Any:
        """输出一段模型摘要，直接贴进文档的「为什么选这个模型」一节。"""
        p = self.count_params()
        lines = ["模型结构：%s" % self.arch,
                 "输入尺寸：%d x %d" % (self.input_size, self.input_size),
                 "取样尺度：P%s  步长 %s  通道 %s"
                 % ("/".join(str(i) for i in self.out_stages or []),
                    "/".join(str(s) for s in self.out_strides or []),
                    "/".join(str(c) for c in self.out_channels or [])),
                 "特征图尺寸：%s" % (self.feat_hw,),
                 "检测头每层输出通道：%s（4 回归 + %d 类别%s）"
                 % (self.head_out_channels, self.num_classes,
                    " + 1 置信度" if self.has_objectness else ""),
                 "参数量：合计 %d（%.3f M），主干 %d，检测头 %d"
                 % (p["total"], p["million"], p["backbone"], p["head"])]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 损失函数类
# ---------------------------------------------------------------------------
class LossFunction:
    """三项损失，一项一个函数：框回归 + 置信度 + 分类。

    检测头每个格点吐 11 个数，三个损失各管其中一段，互不重叠：

        通道 0 : 4         box_loss             框回归，1 - CIoU
        通道 4 : 4+nc      classification_loss  每类一个 BCE，只在正样本上算
        通道 4+nc          objectness_loss      有没有目标，全部格点都算，带忽略区

    为什么必须是三项、不能合成一项：
        * 输出空间不同。前 4 个数是几何量，后 7 个数是概率（未过 sigmoid 的
          分数）。几何量只能跟几何量比（重叠面积），概率只能跟概率比（似然），
          混成一个损失没有物理意义。
        * 正负样本的构成不同。分类只该在正样本上学——背景的类别是"未知"，
          不是"6 个 0"，让背景参与分类等于把分类头往全零压。置信度必须
          在全部 8400 个格点上学，因为它要学的就是分辨背景。
        * 分开记录才能诊断。总损失在降、某一项在涨，说明权重失衡；只记
          总和看不出这件事。

    输入约定（训练循环要按这个格式喂）:
        targets  长度为 batch 的列表，每项是 [n, 5] 张量，五列依次为
                 (类别下标, 中心横, 中心纵, 宽, 高)。坐标单位是**输入尺度的
                 像素**（已做过 letterbox，落在 0~input_size），既不是原图
                 像素，也不是归一化到 0~1 的值。
        raws     tuple，三层原始输出，每层 [B, 4+nc+1, H, W]，没解码、
                 没过 sigmoid（分类损失要用 logits 算，过了 sigmoid 就不对）。
    """

    def __init__(self, config: Any = None, num_classes: int = 6,
                 head: Any = None, box_gain: float = 7.5,
                 obj_gain: float = 1.0, cls_gain: float = 0.5,
                 ignore_iou: float = 0.5, neighbor: int = 1) -> None:
        """head 传检测头（build_head 的返回值）：解码只保留它那一份公式。

        三个 gain 是起始值，不是定论。调法不看绝对值看比例：跑完第一个
        epoch 把三个子项打印出来，谁比别的差一个数量级以上就调谁，目标是
        三者落在同一量级（比如都在 0.2~2 之间）。

        neighbor=1 表示正样本取"中心那一格 + 周围 8 格"共 9 个格点。
        """
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes 要是正整数：%r" % (num_classes,))
        if not isinstance(neighbor, int) or neighbor < 0:
            raise ValueError("neighbor 要是 >=0 的整数：%r" % (neighbor,))
        for name, g in (("box_gain", box_gain), ("obj_gain", obj_gain),
                        ("cls_gain", cls_gain)):
            if not isinstance(g, (int, float)) or g < 0:
                raise ValueError("%s 要是非负数：%r" % (name, g))

        self.config = config
        self.name = "ciou + bce(obj) + bce(cls)"
        self.num_classes = int(num_classes)
        self.head = head
        self.head_num_classes = None        # 建好时从 head 读一次，防两边不一致
        self.class_weights = None           # 长度为 nc 的列表，None 表示不加权
        self.component_weights = {"box": float(box_gain),
                                  "obj": float(obj_gain),
                                  "cls": float(cls_gain)}
        self.ignore_iou = float(ignore_iou)
        self.neighbor = int(neighbor)
        self.reduction = "sum / 正样本数"
        self.history = []                   # 逐 step 的 {"box","obj","cls","total"}
        self._torch = None
        self._nn = None

    # ---- 4.0 延迟导入 torch（与 ModelBuilder 同一套做法） -------------------
    def _import_torch(self) -> Tuple[Any, Any]:
        """只用损失时才导入 torch，import train 本身不受影响。"""
        if self._torch is not None:
            return self._torch, self._nn
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "没装 PyTorch，算不了损失。现场无 GPU，装 CPU 版就够：\n"
                "  python3 -m pip install --no-cache-dir torch "
                "--index-url https://download.pytorch.org/whl/cpu") from exc
        self._torch, self._nn = torch, nn
        return torch, nn

    # ---- 4.1 配对：决定每个格点是正样本、负样本还是忽略样本 ----------------
    def match_targets(self, preds: Any, targets: Any) -> Any:
        """把预测框和真值框配对，给每个格点标上身份。

        三条规则（规则一是解码公式的逆运算，不是拍脑袋定的）：

            规则一 中心归属。真值框中心落在哪一格，那一格负责这个框。
                  依据：解码用的 cx=(格下标+2*sigmoid(t)-0.5)*步长，sigmoid
                  值域 (0,1)，所以一个格点只能表示中心在自己这格、以及
                  左右各半格范围内的框。让中心落在它格内的目标归它管，
                  是唯一自洽的分配。
            规则二 3x3 邻域。中心格加周围 8 格一起当正样本，回归目标都是
                  同一个真值框。理由：8400 格点里每个目标只有 1 个正样本的话
                  信号太稀；9 个相当于把监督密度放大 9 倍。
            规则三 尺度分层。真值框按 min(宽,高)/4 归到 {8,16,32} 里最接近
                  的那一层，只有那一层参与；另外两层对这一格仍标负样本。

        参数:
            preds    head.decode(raws)[0]，即 [B, N, 4] 的 xyxy（输入尺度像素）
            targets  按上面的输入约定
        返回:
            list，每层一个 dict：
                level     第几层（0/1/2）
                off       这一层在 N 个格点里的起始下标
                flat      正样本格点在本层内的展开下标 [P]
                img       正样本属于第几张图 [P]
                gt        [P] 对应 targets 里的第几个框（第几行）
                idx       img * N + off + flat，用来一次取出预测框 [P]
                pos_mask  [B, H*W] 布尔，哪些格点是正样本
        """
        torch, _ = self._import_torch()
        if self.head is None:
            raise RuntimeError(
                "LossFunction 还没拿到检测头：构造时传 head=model.head，"
                "或建好模型后赋 loss_fn.head = model.head")

        head = self.head
        feat_hw = [(int(h), int(w)) for h, w in head.feat_hw]
        strides = [int(s) for s in head.strides]
        sizes = [h * w for h, w in feat_hw]
        offsets = [sum(sizes[:k]) for k in range(len(sizes))]

        b, n_all, _ = preds.shape
        device = preds.device
        gts = [torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 5)
               for t in targets]
        if len(gts) != b:
            raise ValueError("targets 有 %d 张图，预测有 %d 张，对不上"
                             % (len(gts), b))

        matched = []
        for k, (h, w) in enumerate(feat_hw):
            flat, img, gt = [], [], []
            for bi, one in enumerate(gts):
                for gi in range(one.shape[0]):
                    gx, gy = float(one[gi, 1]), float(one[gi, 2])
                    gw, gh = float(one[gi, 3]), float(one[gi, 4])
                    # 规则三：先判这个框该归哪一层，不是本层就跳过
                    scale = min(gw, gh) / 4.0
                    lv = min(range(len(strides)),
                             key=lambda t: abs(strides[t] - scale))
                    if lv != k:
                        continue
                    s = float(strides[k])
                    # 规则一：中心格的整数下标
                    j = min(max(int(gx // s), 0), w - 1)
                    i = min(max(int(gy // s), 0), h - 1)
                    # 规则二：中心格 + 周围 neighbor 圈
                    for di in range(-self.neighbor, self.neighbor + 1):
                        for dj in range(-self.neighbor, self.neighbor + 1):
                            ii, jj = i + di, j + dj
                            if 0 <= ii < h and 0 <= jj < w:
                                flat.append(ii * w + jj)
                                img.append(bi)
                                gt.append(gi)

            flat_t = torch.as_tensor(flat, dtype=torch.long, device=device)
            img_t = torch.as_tensor(img, dtype=torch.long, device=device)
            gt_t = torch.as_tensor(gt, dtype=torch.long, device=device)
            pos_mask = torch.zeros(b, h * w, dtype=torch.bool, device=device)
            if flat_t.numel():
                pos_mask[img_t, flat_t] = True
            matched.append({
                "level": k, "off": offsets[k], "flat": flat_t,
                "img": img_t, "gt": gt_t,
                "idx": img_t * n_all + offsets[k] + flat_t,
                "pos_mask": pos_mask,
            })
        return matched

    # ---- 4.2 总损失：三项加权求和，各自除以正样本数 ------------------------
    def compute(self, preds: Any, targets: Any) -> Any:
        """算出总损失，并把三个子项一起返回。

        参数:
            preds    raws，三层原始输出（不是 decode 之后的）
            targets  按上面的输入约定
        返回:
            (总损失, {"box": .., "obj": .., "cls": ..})，四个都是标量张量

        归一化除以"正样本数"而不是 batch 大小：一张图可能 0 个目标、也可能
        6 个，除以图片张数会让目标多的图梯度大、空图梯度小，同一批里极不
        稳定。除以正样本数等于算"平均每个目标的损失"，与图里有几个目标无关。
        """
        torch, _ = self._import_torch()
        if self.head is None:
            raise RuntimeError("LossFunction 还没拿到检测头：loss_fn.head = model.head")

        if self.head_num_classes is None:
            self.head_num_classes = int(self.head.num_classes)
            if self.head_num_classes != self.num_classes:
                raise ValueError("类别数不一致：损失这边 %d，检测头那边 %d"
                                 % (self.num_classes, self.head_num_classes))

        boxes = self.head.decode(preds)[0]          # [B, N, 4] 解码只做一次
        matched = self.match_targets(boxes, targets)

        n_pos = sum(int(m["flat"].numel()) for m in matched)
        denom = float(max(n_pos, 1))                # 空图时也要能算（全 0）

        # 三个损失各自返回"总和"，除正样本数统一在这里做，避免三处各除一次
        l_box = self.box_loss(matched, boxes, targets) / denom
        l_obj = self.objectness_loss(matched, preds, boxes, targets) / denom
        l_cls = self.classification_loss(matched, preds, targets) / denom

        w = self.component_weights
        total = w["box"] * l_box + w["obj"] * l_obj + w["cls"] * l_cls
        parts = {"box": l_box.detach(), "obj": l_obj.detach(),
                 "cls": l_cls.detach()}
        return total, parts

    # ---- 4.3 分类损失：逐类 BCE，只在正样本上算 ----------------------------
    def classification_loss(self, matched: Any, raws: Any, targets: Any) -> Any:
        """6 个类别各算一个二元交叉熵，只在正样本格点上算，返回总和。

        为什么是 BCE 而不是 softmax + 交叉熵：
            * 能表达重叠目标。softmax 强制 6 个概率和为 1，表达不了"这一格
              同时是书和笔"，而你的数据里笔压在书上是真实存在的。
            * 能单独给某一类加权。书 213 个框、杯子只有 60 个，BCE 逐类
              独立加权；softmax 里改一个类的权重会连带改变其他类的概率。
            * 梯度互不污染。softmax 的梯度跟所有类的分数差有关，一类异常
              会波及全部；BCE 各算各的。

        为什么只在正样本上算：背景格点的类别是"未知"，不是"6 个 0"。
        让它们参与等于告诉模型"背景一定不是任何一类"，分类头会被持续
        往全零压。分辨有没有目标这件事交给 objectness_loss。

        梯度性质（写报告时可引）：单类 BCE 对 logit 的导数是 p - y，
        因为 sigmoid 导数恰好约掉对数里的分母。所以梯度大小就是"预测概率
        减真值"：预测 0.9、真值 1 轻轻推，预测 0.1、真值 1 狠狠推。
        """
        torch, nn = self._import_torch()
        nc = self.num_classes
        total = torch.zeros((), device=raws[0].device)
        weight = None
        if self.class_weights is not None:
            weight = torch.as_tensor(self.class_weights, dtype=torch.float32,
                                     device=raws[0].device)

        for m in matched:
            if m["flat"].numel() == 0:
                continue
            k = m["level"]
            raw = raws[k]
            b, _, h, w = raw.shape
            # [B, nc, H, W] -> [B*H*W, nc]，行号是 img*(H*W)+flat，
            # 注意这个行号和 boxes 的 idx（img*N+off+flat）不是同一个编号空间
            logits = raw[:, 4:4 + nc].permute(0, 2, 3, 1).reshape(b * h * w, nc)
            sel = logits[m["img"] * (h * w) + m["flat"]]

            # 真值：正样本负责哪个框，就取那个框的类别，做成 one-hot
            cls_idx = torch.cat([
                torch.as_tensor(targets[bi], dtype=torch.float32,
                                device=raw.device).reshape(-1, 5)[g, 0:1]
                for bi, g in zip(m["img"].tolist(), m["gt"].tolist())
            ]).long().reshape(-1)
            target = nn.functional.one_hot(cls_idx, nc).to(sel.dtype)

            per = nn.functional.binary_cross_entropy_with_logits(
                sel, target, reduction="none")
            if weight is not None:
                per = per * weight.reshape(1, nc)
            total = total + per.sum()
        return total

    # ---- 4.4 框回归损失：1 - CIoU，只在正样本上算 --------------------------
    def box_loss(self, matched: Any, boxes: Any, targets: Any) -> Any:
        """框回归用 1 - CIoU，只在正样本格点上算，返回总和。

        为什么不用均方误差（三个致命问题）:
            1. 它假设四个坐标的误差互相独立，实际宽高高度相关（框整体大了，
               宽高一起偏）。
            2. 它对尺度不敏感。大框偏 10 像素几乎还重叠，小框偏 10 像素完全
               不重叠，MSE 给一样的惩罚。你的书可能 400 像素宽、笔只有 15
               像素宽，小目标的误差会被大目标彻底淹没。
            3. 它优化的不是评价指标。最后评测用的是重叠面积比，MSE 小不代表
               重叠面积比高。

        为什么是 CIoU 而不是直接用 1 - IoU:
            * 两框不相交时 IoU 恒为 0、损失恒为 1，梯度是 0，训练初期学不动。
              GIoU 减去"最小外接框里的空白占比"，让不相交时也有下降方向。
            * GIoU 分不出重叠方式（居中重叠和偏在一角卡着同分），也看不出
              长宽比是否离谱。CIoU 再加中心距惩罚和长宽比惩罚：

                  CIoU = IoU - d^2/c^2 - alpha*v
                  d^2   两框中心距平方
                  c^2   最小外接矩形对角线平方
                  v     长宽比一致性（用反正切把无界的比值压成有界角度）
                  alpha 自动权重 v/(1-IoU+v)

        实现要求：必须在**像素坐标**上算，不能直接在回归的四个原始数上算。
        那四个数是取过对数的相对量，物理含义不同，算出来的数没有意义。
        """
        torch, _ = self._import_torch()
        nc = self.num_classes

        pred_list, gt_list = [], []
        for m in matched:
            if m["flat"].numel() == 0:
                continue
            # 预测框：boxes 是 [B, N, 4]，按 idx 一次取出所有正样本
            pred_list.append(boxes.reshape(-1, 4)[m["idx"]])
            # 真值框：按 (第几张图, 第几个框) 一一取出
            rows = []
            for bi, g in zip(m["img"].tolist(), m["gt"].tolist()):
                one = torch.as_tensor(targets[bi], dtype=torch.float32,
                                      device=boxes.device).reshape(-1, 5)
                rows.append(one[g, 1:5])
            gt_list.append(torch.stack(rows))

        if not pred_list:
            return torch.zeros((), device=boxes.device)

        pred = torch.cat(pred_list, dim=0)
        gt = self._to_xyxy(torch.cat(gt_list, dim=0))   # 真值是中心式，先转左上右下

        eps = 1e-7
        # ---- 交集 / 并集 ----
        iw = (torch.min(pred[:, 2], gt[:, 2]) - torch.max(pred[:, 0], gt[:, 0])).clamp(min=0)
        ih = (torch.min(pred[:, 3], gt[:, 3]) - torch.max(pred[:, 1], gt[:, 1])).clamp(min=0)
        inter = iw * ih
        pw = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
        ph = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
        gw = (gt[:, 2] - gt[:, 0]).clamp(min=eps)
        gh = (gt[:, 3] - gt[:, 1]).clamp(min=eps)
        union = pw * ph + gw * gh - inter + eps
        iou = inter / union

        # ---- 中心距惩罚 ----
        d2 = ((pred[:, 0] + pred[:, 2] - gt[:, 0] - gt[:, 2]) ** 2 +
              (pred[:, 1] + pred[:, 3] - gt[:, 1] - gt[:, 3]) ** 2) / 4.0
        cw = torch.max(pred[:, 2], gt[:, 2]) - torch.min(pred[:, 0], gt[:, 0])
        ch = torch.max(pred[:, 3], gt[:, 3]) - torch.min(pred[:, 1], gt[:, 1])
        c2 = cw ** 2 + ch ** 2 + eps

        # ---- 长宽比惩罚：ratio 无界，用反正切压到 (0, pi/2) ----
        v = (4.0 / (3.141592653589793 ** 2)) * (
            torch.atan(pw / ph) - torch.atan(gw / gh)) ** 2
        alpha = v / (1.0 - iou + v + eps)

        ciou = iou - d2 / c2 - alpha * v
        return (1.0 - ciou).sum()

    # ---- 4.5 置信度损失：BCE，全部格点，带忽略区 --------------------------
    def objectness_loss(self, matched: Any, raws: Any, boxes: Any,
                        targets: Any) -> Any:
        """置信度用 BCE，8400 个格点全算，但剔除忽略区，返回总和。

        真值三档:
            正样本格点（匹配上的）           -> 1
            负样本格点（其余）               -> 0
            忽略格点                         -> 不参与

        忽略区为什么必须有：每个真值框只有 9 个正样本名额，但可能有几十个
        格点都预测出了不错的框。剩下那些框定位其实对，只是没抢到名额。把它们
        当负样本惩罚，等于告诉模型"你框得对，但我要你放弃它"，梯度会跟正样本
        打架，收敛明显变慢。判据：自己预测的框跟任一真值框重叠度 > ignore_iou
        且没被匹配上。

        梯度性质与分类分支相同（p - y），但这里 p 是从 8400 个格点里算出来的，
        正负比例悬殊（一张 5 个目标的图约 45:8355），所以这一项天然是主导项，
        起始权重给 1.0 即可，不要放大。
        """
        torch, nn = self._import_torch()
        nc = self.num_classes
        total = torch.zeros((), device=raws[0].device)

        for m in matched:
            k = m["level"]
            raw = raws[k]
            b, _, h, w = raw.shape
            logits = raw[:, 4 + nc:].permute(0, 2, 3, 1).reshape(b * h * w)
            target = m["pos_mask"].reshape(-1).to(logits.dtype)

            # 忽略区：本层每个格点的预测框 vs 这张图的全部真值框
            seg = boxes[:, m["off"]:m["off"] + h * w]      # [B, HW, 4]
            keep = torch.ones(b, h * w, dtype=torch.bool, device=logits.device)
            for bi in range(b):
                one = torch.as_tensor(targets[bi], dtype=torch.float32,
                                      device=logits.device).reshape(-1, 5)
                if one.shape[0] == 0:
                    continue
                iou = self._iou_matrix(seg[bi], self._to_xyxy(one[:, 1:5]))
                best = iou.max(dim=1).values                  # [HW]
                ign = (best > self.ignore_iou) & (~m["pos_mask"][bi])
                keep[bi] = ~ign

            per = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            total = total + (per * keep.reshape(-1).to(per.dtype)).sum()
        return total

    # ---- 4.6 小工具：坐标换算与两两 IoU ------------------------------------
    def _to_xyxy(self, b: Any) -> Any:
        """把 (中心横, 中心纵, 宽, 高) 换成 (左上横, 左上纵, 右下横, 右下纵)。

        真值按 centerness 格式给（跟标注工具一致），但 IoU/CIoU 必须在
        左上右下格式上算，这一步不能省，否则算出来的重叠度是错的。
        """
        torch, _ = self._import_torch()
        half_w = b[:, 2] * 0.5
        half_h = b[:, 3] * 0.5
        return torch.stack([b[:, 0] - half_w, b[:, 1] - half_h,
                            b[:, 0] + half_w, b[:, 1] + half_h], dim=-1)

    def _iou_matrix(self, a: Any, b: Any) -> Any:
        """a 是 [N,4]，b 是 [M,4]，返回 [N,M] 的 IoU 矩阵（xyxy 像素坐标）。"""
        torch, _ = self._import_torch()
        eps = 1e-7
        ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        iw = (torch.min(ax2.unsqueeze(1), bx2.unsqueeze(0)) -
              torch.max(ax1.unsqueeze(1), bx1.unsqueeze(0))).clamp(min=0)
        ih = (torch.min(ay2.unsqueeze(1), by2.unsqueeze(0)) -
              torch.max(ay1.unsqueeze(1), by1.unsqueeze(0))).clamp(min=0)
        inter = iw * ih
        area_a = ((ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)).unsqueeze(1)
        area_b = ((bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)).unsqueeze(0)
        return inter / (area_a + area_b - inter + eps)

    # ---- 4.7 长尾类加权 -----------------------------------------------------
    def set_class_weights(self, distribution: Any) -> Any:
        """按各类框数的反比设权重，均值归一化到 1，供 §3 长尾对比实验。

        distribution 可以是长度 nc 的列表（各类框数），也可以是 {类别名: 数量}。
        权重公式 w_i = 总数 / (nc * 该类数量)，再整体除以均值。

        必须提醒：加权一定提高稀有类的召回、同时压低精确率，是不是净收益
        只能用验证集测，不能预设。所以默认不加权（None），这组实验单独跑。
        """
        if isinstance(distribution, dict):
            counts = [float(v) for v in distribution.values()]
        else:
            counts = [float(v) for v in distribution]
        if len(counts) != self.num_classes:
            raise ValueError("给了 %d 个类的分布，但 num_classes=%d"
                             % (len(counts), self.num_classes))
        total = sum(counts)
        if total <= 0:
            raise ValueError("类别分布总数要大于 0：%r" % (distribution,))
        raw = [total / (len(counts) * max(c, 1.0)) for c in counts]
        mean = sum(raw) / len(raw)
        self.class_weights = [w / mean for w in raw]
        return self.class_weights

    # ---- 4.8 记录 -----------------------------------------------------------
    def log(self, record: Any = None) -> Any:
        """把一次 step 的三个子项存下来，供 OutputExporter 画曲线。

        record 传 {"box":.., "obj":.., "cls":.., "total":..}（标量或 float）。
        不传则只返回已存的记录列表。三个子项分开记是刻意的：总分在降但某一
        项在涨，是权重失衡的直接证据，只记总和看不出来。
        """
        if record is not None:
            row = {}
            for key, val in dict(record).items():
                row[key] = float(val.detach()) if hasattr(val, "detach") else float(val)
            self.history.append(row)
        return self.history


# ---------------------------------------------------------------------------
# 5. 优化算法类
# ---------------------------------------------------------------------------
class OptimizerBuilder:
    """构建优化器与学习率策略，管理每次参数更新。

    这里管两件不同的事，先分开说清，否则容易混成一个概念：

        * 优化器（AdamW）：拿到梯度之后，每个参数往哪走、各走多远。
        * 学习率曲线（预热 + 余弦）：这个"走多远"的基准值随训练进程怎么变。

    为什么是 AdamW 而不是 Adam：Adam 的 weight_decay 是把衰减项加进梯度里、
    再被自适应步长除掉，实际效果偏弱且不可控；AdamW 把衰减解耦成独立的一步，
    才真的是 L2 正则。代码只差一个字母，行为不同。

    为什么学习率从 1e-3 起：Adam 的自适应步长把每个参数的更新量归一到"约等于
    学习率"的量级，所以它用的学习率比普通随机梯度下降小一个数量级。照抄 SGD
    的 1e-2 会直接把权重打飞。

    为什么参数要分两组：批归一化的缩放和偏移、所有偏置，这三类不承载"表达
    能力"，衰减它们只会把归一化偏移掉。规则很简单——张量维度小于 2 的参数
    不衰减（偏置、BN 的 gamma/beta 都是一维）。

    题面 §4 要求学习率策略留档：state_summary() 出静态超参，
    history 出逐轮实际学习率，两者一起写进 docs/train.md。
    """

    def __init__(self, config: Any = None, lr: float = 1e-3,
                 weight_decay: float = 1e-3, betas: Any = (0.9, 0.999),
                 eps: float = 1e-8, batch_size: int = 16,
                 total_epochs: int = 150, warmup_epochs: float = 3,
                 min_lr_ratio: float = 0.01, grad_clip: float = 10.0,
                 scheduler_name: str = "cosine") -> None:
        """默认值就是那组起始配置，只把 lr 当唯一需要扫的参数。

        参数:
            lr            初始学习率（终点值是 lr * min_lr_ratio）
            weight_decay  只作用在"维度 >= 2"的参数上
            betas         Adam 的一阶矩/二阶矩衰减率；(0.9, 0.999) 基本不用动
            eps           防除零的常数；用半精度训练时要提到 1e-4 防下溢
            batch_size    只用于推算一轮有多少步、预热折算成多少步，不改变优化器
            total_epochs  总轮数，余弦曲线按它铺满
            warmup_epochs 预热轮数，前几轮从 lr/warmup 线性升到 lr
            min_lr_ratio  余弦终点占初始学习率的比例
            grad_clip     梯度整体范数的上限，0 表示不裁剪
            scheduler_name "cosine"（默认）或 "plateau"（按验证集指标降 lr）
        """
        # ---- 参数校验：错的值要在这里炸，不要等到训练第 40 轮 ----
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr 要是正数：%r" % (lr,))
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError("weight_decay 要是非负数：%r" % (weight_decay,))
        if len(tuple(betas)) != 2 or not all(0 <= float(b) < 1 for b in betas):
            raise ValueError("betas 要是两个 [0,1) 之间的数：%r" % (betas,))
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("eps 要是正数：%r" % (eps,))
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size 要是正整数：%r" % (batch_size,))
        if not isinstance(total_epochs, int) or total_epochs < 1:
            raise ValueError("total_epochs 要是正整数：%r" % (total_epochs,))
        if not isinstance(warmup_epochs, (int, float)) or warmup_epochs < 0:
            raise ValueError("warmup_epochs 要是非负数：%r" % (warmup_epochs,))
        if not (0 <= warmup_epochs < total_epochs):
            raise ValueError("warmup_epochs 要小于 total_epochs：%r vs %r"
                             % (warmup_epochs, total_epochs))
        if not (0 <= min_lr_ratio <= 1):
            raise ValueError("min_lr_ratio 要在 0~1 之间：%r" % (min_lr_ratio,))
        if not isinstance(grad_clip, (int, float)) or grad_clip < 0:
            raise ValueError("grad_clip 要是非负数：%r" % (grad_clip,))
        if scheduler_name not in ("cosine", "plateau"):
            raise ValueError("scheduler_name 只支持 'cosine' / 'plateau'：%r"
                             % (scheduler_name,))

        self.config = config
        self.name = "AdamW"
        self.lr = float(lr)
        self.momentum = None                # AdamW 不用动量，一阶矩由 betas[0] 承担
        self.weight_decay = float(weight_decay)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.batch_size = int(batch_size)
        self.total_epochs = int(total_epochs)
        self.warmup = float(warmup_epochs)
        self.min_lr_ratio = float(min_lr_ratio)
        self.grad_clip = float(grad_clip)
        self.scheduler_name = scheduler_name
        self.optimizer = None
        self.scheduler = None
        self.param_groups = None            # {衰减组参数名, 不衰减组参数名}
        self.epoch = 0                      # 已经推进过几轮，从 0 开始
        self.last_grad_norm = None          # 上一次裁剪前的梯度范数，用于诊断
        self.history = []                   # [(轮次, 学习率)]，画曲线用
        self._params = None                 # 参与训练的扁平参数表，裁剪要用
        self._torch = None
        self._nn = None

    # ---- 5.1 延迟导入 torch（与前面两个类同一套做法） ----------------------
    def _import_torch(self) -> Tuple[Any, Any]:
        """只用优化器时才导入 torch，import train 本身不受影响。"""
        if self._torch is not None:
            return self._torch, self._nn
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "没装 PyTorch，建不了优化器。现场无 GPU，装 CPU 版就够：\n"
                "  python3 -m pip install --no-cache-dir torch "
                "--index-url https://download.pytorch.org/whl/cpu") from exc
        self._torch, self._nn = torch, nn
        return torch, nn

    # ---- 5.2 建优化器：按维度分两组 ---------------------------------------
    def build(self, model: Any) -> Any:
        """创建 AdamW 并绑定模型参数，返回优化器。

        model 可以是整个检测模型，也可以只传主干（测试时方便）。
        只收 requires_grad=True 的参数，冻结的部分会被自动排除。
        """
        torch, _ = self._import_torch()
        if model is None:
            raise ValueError("build 收到的是 None，先把模型建好再传进来")
        if not hasattr(model, "named_parameters"):
            raise ValueError("传进来的对象没有 named_parameters，不像是一个模型")

        decay_names, decay, no_decay_names, no_decay = [], [], [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:                 # 卷积核、全连接矩阵：要衰减
                decay.append(p)
                decay_names.append(pname)
            else:                           # 偏置、BN 的 gamma/beta：不衰减
                no_decay.append(p)
                no_decay_names.append(pname)

        if not decay and not no_decay:
            raise ValueError("模型里没有任何需要训练的参数（全部被冻结了？）")

        groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(groups, lr=self.lr,
                                           betas=self.betas, eps=self.eps)
        self._params = decay + no_decay
        self.param_groups = {"衰减组": decay_names, "不衰减组": no_decay_names}
        self.name = ("AdamW(lr=%.1e, wd=%.1e, betas=%.3f/%.3f, 不衰减参数 %d 个)"
                     % (self.lr, self.weight_decay, self.betas[0], self.betas[1],
                        len(no_decay)))
        return self.optimizer

    # ---- 5.3 建学习率曲线：预热 + 余弦 ------------------------------------
    def set_scheduler(self) -> Any:
        """创建学习率策略，按"轮"推进（不是按步）。

        预热段：第 0 轮用 lr/warmup，线性升到 lr。为什么不能从 0 或全速开始：
            Adam 的二阶矩估计在最初几步是从零累积的，更新量严重偏大；同时批
            归一化的滑动统计量是拿一两个批次估出来的，很不准。两者叠加，一上来
            全速容易把权重推进坏区域，之后回不来。
        余弦段：从 lr 平滑降到 lr * min_lr_ratio。比阶梯下降稳，因为学习率
            不跳变。

        plateau 模式（验证集不涨就减半）不走预热：两者混用会让"哪一轮该是多少"
        说不清，所以设了预热就禁止选 plateau，反之亦然。
        """
        torch, _ = self._import_torch()
        if self.optimizer is None:
            raise RuntimeError("先调用 build() 建优化器，再建学习率策略")

        if self.scheduler_name == "plateau":
            if self.warmup > 0:
                raise ValueError(
                    "plateau 模式不支持预热，请把 warmup_epochs 设为 0，"
                    "或者继续用 cosine")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=10,
                min_lr=self.lr * self.min_lr_ratio)
            self.name += " + plateau(factor=0.5, patience=10)"
        else:
            total = float(self.total_epochs)
            warm = float(self.warmup)
            low = float(self.min_lr_ratio)

            def factor(e: float) -> float:
                """给定轮次返回学习率的倍率，值域 [min_lr_ratio, 1]。"""
                if warm > 0 and e < warm:
                    return (e + 1.0) / warm          # 第 0 轮 = 1/warm，不用 0
                if total <= warm:
                    return 1.0
                progress = (e - warm) / (total - warm)
                progress = min(max(progress, 0.0), 1.0)
                return low + (1.0 - low) * 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=factor)
            self.name += (" + warmup%.0fep + cosine->%.0f%%"
                          % (warm, low * 100))
        self.history.append((0, self.current_lr()))
        return self.scheduler

    # ---- 5.4 清梯度 --------------------------------------------------------
    def zero_grad(self) -> None:
        """清空梯度。set_to_none=True 直接把梯度置空，省一次全参数写内存。"""
        if self.optimizer is None:
            raise RuntimeError("还没 build()，没有优化器可清")
        self.optimizer.zero_grad(set_to_none=True)

    # ---- 5.5 参数更新：先裁剪，再走一步 -----------------------------------
    def step(self) -> None:
        """把梯度整体裁到上限，然后更新参数。

        为什么要裁剪：置信度损失是在 8400 个格点上求和再除以几十个正样本，
        早期梯度量级本来就不稳，一个"目标特别多"的批次就能把权重打飞。
        裁剪只在超限时等比缩小，正常梯度不受影响。
        """
        if self.optimizer is None:
            raise RuntimeError("还没 build()，没有优化器可更新")
        if self.grad_clip and self.grad_clip > 0:
            torch, _ = self._import_torch()
            total_norm = torch.nn.utils.clip_grad_norm_(
                self._params, self.grad_clip)
            self.last_grad_norm = float(total_norm)
        else:
            self.last_grad_norm = None
        self.optimizer.step()

    # ---- 5.6 推进学习率（每轮一次） ---------------------------------------
    def scheduler_step(self, metric: Any = None) -> None:
        """每轮结束时推进一步学习率。

        metric 只有在 plateau 模式下才需要，而且**只能传验证集指标**：
        题面 §10 明确禁止用测试集 mAP 选模型或调参，拿测试集指标驱动学习率
        就是同一种作弊，只是更隐蔽。
        """
        if self.scheduler is None:
            raise RuntimeError("还没 set_scheduler()，没有学习率曲线可推进")
        if self.scheduler_name == "plateau":
            if metric is None:
                raise ValueError("plateau 模式必须传验证集指标（比如 mAP）")
            self.scheduler.step(float(metric))
        else:
            self.scheduler.step()
        self.epoch += 1
        self.history.append((self.epoch, self.current_lr()))

    # ---- 5.7 当前学习率 ----------------------------------------------------
    def current_lr(self) -> Any:
        """返回当前学习率（各组相同，取第一组）。写进每轮日志与曲线。"""
        if self.optimizer is None:
            return None
        return float(self.optimizer.param_groups[0]["lr"])

    # ---- 5.8 超参摘要 ------------------------------------------------------
    def state_summary(self) -> Any:
        """导出全部超参，供 docs/train.md 与"一键复现"核对。

        和 history（逐轮学习率）配合：一个答"配置是什么"，一个答"曲线长什么样"。
        """
        steps_per_epoch = None              # 要样本数才知道，训练循环算完填进去
        return {
            "optimizer": self.name,
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "no_decay_param_count": len(self.param_groups["不衰减组"])
            if self.param_groups else None,
            "decay_param_count": len(self.param_groups["衰减组"])
            if self.param_groups else None,
            "batch_size": self.batch_size,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup,
            "min_lr_ratio": self.min_lr_ratio,
            "grad_clip": self.grad_clip,
            "scheduler": self.scheduler_name,
            "epoch_done": self.epoch,
            "last_grad_norm": self.last_grad_norm,
            "steps_per_epoch": steps_per_epoch,
        }


# ---------------------------------------------------------------------------
# 6. 全体迭代类：迭代终止条件等迭代要求
# ---------------------------------------------------------------------------
class TrainingLoop:
    """把前面五个类串成 epoch 迭代，负责验证、早停、留档。

    早停判据（这个类的核心）：

        每一轮验证结束后算一次「验证集损失的相对提高率」

            提高率 = (参考损失 − 本轮损失) / 参考损失

        参考损失是上一次「被认可」的验证损失。当提高率小于 min_delta 时，
        这一轮记为「没有实质进步」；连续 patience 轮没有实质进步就停止，
        并记下停止原因，供报告里的「迭代终止条件」一节直接引用。

    三个设计上的取舍，写报告时都要交代：

        * 为什么用相对提高率，不用绝对差值。损失从 2.0 降到 1.0 是 50%，
          从 0.02 降到 0.01 也是 50%。同一个绝对阈值（比如 0.001）在训练
          早期等于没有要求、在末期等于永不满足；相对值无量纲，早晚期同一个
          数才有同样的含义。

        * 为什么参考值不随每轮刷新。如果每轮都拿上一轮当基准，那么 0.1%
          的抖动也算「进步」，计数器永远清零，永远不会停。拿「上次被认可的
          损失」当基准，条件是「相对它的提升达到阈值」，这才是「训练已经没有
          实质收益」。

        * 为什么还要 patience。验证损失是单轮统计量，本身带抖动。patience=1
          等于「一超阈值立刻停」，patience=30 是「连着 30 轮都没实质提升才
          认输」。两者严格程度差 30 倍，报告里必须写明用的哪个。

    另外三条与题面直接相关的约束：
        * §10：val 集调参，test 集只在最终报告时用一次。所以这里只碰
          train 和 val，test 的下标原样留着不动。
        * §4：loss 曲线、学习率策略、epoch 数、增强方式全部留档。history
          逐轮记，另落一份 CSV。
        * 现场无 GPU：训练可以在显卡上做，但验证/导出必须能在 CPU 上跑，
          所以 device 默认 "auto"，没有显卡就自动用 CPU，结果一致。
    """

    def __init__(self, config: Any = None, max_epochs: int = 150,
                 patience: int = 30, min_delta: float = 1e-3,
                 batch_size: int = 16, num_workers: int = 0,
                 log_interval: int = 10, ckpt_interval: int = 10,
                 device: str = "auto", out_dir: Optional[str] = None,
                 stop_metric: str = "val_loss", target_metric: Any = None,
                 conf_thres: float = 0.25, nms_iou: float = 0.45,
                 max_det: int = 100, seed: int = 0) -> None:
        """默认值就是那组起始配置，只有 min_delta / patience 需要按数据量调。

        参数:
            max_epochs    最多跑多少轮（硬上限）
            patience      连续多少轮「没实质进步」才停
            min_delta     提高率门槛，0.001 表示「相对提升不足千分之一算没进步」
            batch_size    每批几张图
            stop_metric   用哪个指标判早停：val_loss（默认，越小越好）/
                          mAP50 / mAP50-95（越大越好）
            target_metric 达到这个值就提前收工，None 表示不设
            conf_thres    验证算 mAP 时的分数门槛（训练不受它影响）
            nms_iou       验证算 mAP 时的非极大值抑制阈值
        """
        if not isinstance(max_epochs, int) or max_epochs < 1:
            raise ValueError("max_epochs 要是正整数：%r" % (max_epochs,))
        if not isinstance(patience, int) or patience < 1:
            raise ValueError("patience 要是 >=1 的整数：%r" % (patience,))
        if not isinstance(min_delta, (int, float)) or min_delta < 0:
            raise ValueError("min_delta 要是非负数：%r" % (min_delta,))
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size 要是正整数：%r" % (batch_size,))
        if stop_metric not in ("val_loss", "mAP50", "mAP50-95"):
            raise ValueError("stop_metric 只支持 val_loss / mAP50 / mAP50-95：%r"
                             % (stop_metric,))
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 只支持 'auto' / 'cpu' / 'cuda'：%r" % (device,))

        self.config = config
        self.model = None                   # 来自 ModelBuilder（builder 或容器模块）
        self.loss_fn = None                 # 来自 LossFunction
        self.optimizer = None               # 来自 OptimizerBuilder
        self.datasets = None                # {"X": [...], "Y": [...], "splits": {...}}
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.stop_metric = stop_metric
        self.target_metric = target_metric
        self.min_delta = float(min_delta)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)  # 手动分批，不启多进程，仅与骨架对齐
        self.log_interval = int(log_interval)
        self.ckpt_interval = int(ckpt_interval)
        self.device = device
        self.conf_thres = float(conf_thres)
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)
        self.seed = int(seed)
        self.out_dir = out_dir
        self.epoch = 0
        self.best_metric = None             # 历史最优（保存最优权重用）
        self.best_epoch = None
        self.best_map = None                # 历史最高 mAP50，部署用这一份
        self.best_map_epoch = None
        self.history = []                   # 逐轮记录，画曲线用
        self.should_stop_flag = False
        self.stop_reason = None             # 停止原因，直接写进报告
        self._ref = None                    # 早停参考值
        self._stale = 0                     # 连续没实质进步的轮数
        self._backbone = None
        self._head = None
        self._X = None
        self._Y = None
        self._train_idx = []
        self._val_idx = []
        self._csv_path = None
        self._torch = None
        self._nn = None

    # ---- 6.0 延迟导入 ------------------------------------------------------
    def _import_torch(self) -> Tuple[Any, Any]:
        """只用训练时才导入 torch，import train 本身不受影响。"""
        if self._torch is not None:
            return self._torch, self._nn
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError(
                "没装 PyTorch，训不了。现场无 GPU，装 CPU 版就够：\n"
                "  python3 -m pip install --no-cache-dir torch "
                "--index-url https://download.pytorch.org/whl/cpu") from exc
        self._torch, self._nn = torch, nn
        return torch, nn

    # ---- 6.1 组装：模型 + 数据 + 损失 + 优化器 ----------------------------
    def setup(self) -> None:
        """把零件接起来，建输出目录与 CSV 表头。

        要求的输入:
            self.model      ModelBuilder（用它的 backbone / head），或者任何
                            带 .backbone / .head 两个子模块的容器
            self.loss_fn    LossFunction（要已经挂上 head）
            self.optimizer  OptimizerBuilder（会顺手 build + set_scheduler），
                            也可以直接给 torch 优化器
            self.datasets   {"X": build_dataset 的第一个返回值,
                             "Y": 第二个,
                             "splits": DatasetSplitter.split 的返回值}

        注意 test 集的划分原样保留，这里不碰它（§10）。
        """
        torch, nn = self._import_torch()
        import os as _os

        # ---- 1) 拆出 backbone / head ----
        m = self.model
        if m is None:
            raise ValueError("self.model 还没给")
        backbone = getattr(m, "backbone", None)
        head = getattr(m, "head", None)
        if backbone is None or head is None:
            raise ValueError(
                "self.model 要能取出 .backbone 和 .head 两个子模块："
                "直接传 ModelBuilder（建好之后它有这两个属性），"
                "或传一个装着它们的容器模块")
        self._backbone, self._head = backbone, head

        # ---- 2) 数据 ----
        if not isinstance(self.datasets, dict):
            raise TypeError("self.datasets 要是 {'X','Y','splits'} 字典")
        for key in ("X", "Y", "splits"):
            if key not in self.datasets:
                raise ValueError("self.datasets 缺 %r 这一项" % key)
        self._X = self.datasets["X"]
        self._Y = self.datasets["Y"]
        splits = self.datasets["splits"]
        self._train_idx = list(splits.get("train", []))
        self._val_idx = list(splits.get("val", []))
        if not self._train_idx:
            raise ValueError("train 集是空的，先跑 DatasetSplitter.split()")
        if not self._val_idx:
            raise ValueError("val 集是空的，早停没有依据，没法训练")

        # ---- 3) 设备 ----
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(self.device)
        self._backbone.to(dev)
        self._head.to(dev)

        # ---- 4) 损失要挂 head，否则它解码没有公式可用 ----
        if self.loss_fn is None:
            raise ValueError("self.loss_fn 还没给")
        if getattr(self.loss_fn, "head", None) is None:
            self.loss_fn.head = self._head

        # ---- 5) 优化器 ----
        if self.optimizer is None:
            raise ValueError("self.optimizer 还没给")
        if hasattr(self.optimizer, "build"):        # 来自 OptimizerBuilder
            if getattr(self.optimizer, "optimizer", None) is None:
                # ModuleList 只是拿来凑一个 named_parameters，不参与前向
                holder = nn.ModuleList([self._backbone, self._head])
                self.optimizer.build(holder)
            if getattr(self.optimizer, "scheduler", None) is None:
                self.optimizer.set_scheduler()

        # ---- 6) 输出目录与 CSV ----
        if self.out_dir is None:
            base = _os.path.dirname(_os.path.abspath(__file__))
            self.out_dir = _os.path.join(base, "results", "train")
        _os.makedirs(_os.path.join(self.out_dir, "models"), exist_ok=True)
        _os.makedirs(_os.path.join(self.out_dir, "logs"), exist_ok=True)
        self._csv_path = _os.path.join(self.out_dir, "logs", "history.csv")
        # 每次开训都重写表头：之前是「文件不存在才写」，结果重训时新数据会
        # 追加到上一次的记录后面，两条曲线混在一个文件里，画图完全没法看。
        with open(self._csv_path, "w", encoding="utf-8") as fh:
            fh.write("epoch,train_loss,val_loss,box,obj,cls,mAP50,mAP50-95,"
                     "lr,grad_norm,seconds\n")

        print("[setup] 设备=%s  训练 %d 张 / 验证 %d 张  批大小 %d  "
              "最多 %d 轮  早停 patience=%d  门槛 %.4g"
              % (self.device, len(self._train_idx), len(self._val_idx),
                 self.batch_size, self.max_epochs, self.patience, self.min_delta))

    # ---- 6.2 前向：backbone 出三级特征，head 出三层 raw --------------------
    def _forward(self, imgs: Any) -> Any:
        """image -> 三层 raw 输出。训练和验证共用这一条路径。"""
        return self._head(self._backbone(imgs))

    # ---- 6.3 组一个批次 ---------------------------------------------------
    def _make_batch(self, idxs: Any) -> Any:
        """按下标取图与框，转成网络要的两个张量。

        两处格式转换，都只在这里做，别处不许再改：

            图：uint8 的 [H,W,3] → float32 的 [3,H,W] 且除以 255。
               除 255 是约定，导出 ONNX 时部署端必须做同一件事，否则结果
               差一个数量级——题面 §6 的「预处理一致」指的就是这个。

            框：DataImporter 给的是 [左上横, 左上纵, 右下横, 右下纵, 类别]，
               而损失函数要的是 [类别, 中心横, 中心纵, 宽, 高]。
               两个顺序都不一样（类别从最后一列搬到第一列），直接传过去
               不会报错，只会训出一堆位置错误的框。
        """
        import numpy as np

        torch, _ = self._import_torch()
        imgs = np.stack([self._X[i]["image"] for i in idxs]).astype(np.float32)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous() / 255.0
        imgs = imgs.to(torch.device(self.device))

        targets = []
        for i in idxs:
            rows = []
            for b in self._Y[i]:
                x1, y1, x2, y2, c = (float(b[0]), float(b[1]), float(b[2]),
                                     float(b[3]), float(b[4]))
                rows.append([c, (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1])
            targets.append(torch.tensor(rows, dtype=torch.float32,
                                        device=torch.device(self.device)
                                        ).reshape(-1, 5))
        return imgs, targets

    # ---- 6.4 主循环 -------------------------------------------------------
    def run(self) -> Any:
        """按 epoch 跑：训练 -> 验证 -> 推进学习率 -> 记录 -> 判终止。"""
        import random

        if self._backbone is None:
            self.setup()

        rng = random.Random(self.seed)
        start = self.epoch
        for ep in range(start, self.max_epochs):
            self.epoch = ep

            # 每个 epoch 重排一次训练顺序：数据是按场景分组的，不打乱会出现
            # 「一整批全是同一张桌子」的批次，批归一化的统计会偏
            order = list(self._train_idx)
            rng.shuffle(order)

            tr = self.train_one_epoch(order)
            va = self.validate()

            # 学习率按轮推进；plateau 模式才需要指标，这里统一传 mAP50
            if hasattr(self.optimizer, "scheduler_step"):
                self.optimizer.scheduler_step(va["mAP50"])

            self.log_epoch(tr, va)

            if (ep + 1) % max(self.ckpt_interval, 1) == 0:
                self.save_checkpoint(best=False)

            stop = self.should_stop(tr, va)
            if stop:
                break

        if self.stop_reason is None:
            self.stop_reason = ("跑满 %d 轮（max_epochs），未触发早停"
                                % self.max_epochs)
        print("[训练结束] 已跑 %d 轮，原因：%s"
              % (self.epoch + 1, self.stop_reason))
        return {"history": self.history, "best_metric": self.best_metric,
                "best_epoch": self.best_epoch, "stop_reason": self.stop_reason,
                "epochs": self.epoch + 1}

    # ---- 6.5 训练一轮 -----------------------------------------------------
    def train_one_epoch(self, order: Any = None) -> Any:
        """跑完一轮训练，返回本轮三个子损失与总损失的平均值。

        顺序固定为：清梯度 → 前向 → 算损失 → 反传 → 更新。
        先清再前向，是为了让第一个批次的梯度不会残留上一轮的量。
        """
        torch, _ = self._import_torch()
        self._backbone.train()
        self._head.train()

        order = list(self._train_idx) if order is None else list(order)
        nb = 0
        sums = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
        n_img = 0

        for s in range(0, len(order), self.batch_size):
            chunk = order[s:s + self.batch_size]
            if not chunk:
                continue
            imgs, targets = self._make_batch(chunk)

            self.optimizer.zero_grad()
            raws = self._forward(imgs)
            total, parts = self.loss_fn.compute(raws, targets)
            total.backward()
            self.optimizer.step()

            k = len(chunk)
            sums["loss"] += float(total.detach()) * k
            for key in ("box", "obj", "cls"):
                sums[key] += float(parts[key]) * k
            n_img += k
            nb += 1

            if nb % max(self.log_interval, 1) == 0:
                print("  第 %d 轮 第 %d/%d 批  loss=%.4f (box %.4f obj %.4f cls %.4f)"
                      % (self.epoch + 1, nb, (len(order) + self.batch_size - 1)
                         // self.batch_size, sums["loss"] / max(n_img, 1),
                         sums["box"] / max(n_img, 1), sums["obj"] / max(n_img, 1),
                         sums["cls"] / max(n_img, 1)))

        out = {k: v / max(n_img, 1) for k, v in sums.items()}
        out["grad_norm"] = getattr(self.optimizer, "last_grad_norm", None)
        return out

    # ---- 6.6 验证 ---------------------------------------------------------
    def validate(self) -> Any:
        """在 val 集上算损失与 mAP。

        损失按「每批平均、再按批内图数加权」汇总，等价于逐图平均；
        不按正样本数加权，是为了让这个数跨轮可比（正样本数每轮在变）。

        mAP 同时算 @0.5 和 @0.5:0.95：前者是报告里的主指标，后者用来
        看「框位精度」有没有在提升——两项脱节说明框越来越大或越来越小。
        """
        import numpy as np

        torch, _ = self._import_torch()
        self._backbone.eval()
        self._head.eval()

        sums = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
        n_img = 0
        records = []            # 每张图的预测：(图序号, 类别, 分数, 框)
        gts = []                # 每张图的真值框 [n,4] + 类别

        with torch.no_grad():
            for s in range(0, len(self._val_idx), self.batch_size):
                chunk = self._val_idx[s:s + self.batch_size]
                imgs, targets = self._make_batch(chunk)
                raws = self._forward(imgs)

                total, parts = self.loss_fn.compute(raws, targets)
                k = len(chunk)
                sums["loss"] += float(total.detach()) * k
                for key in ("box", "obj", "cls"):
                    sums[key] += float(parts[key]) * k
                n_img += k

                preds = self._decode_batch(raws)
                for j in range(k):
                    records.append((s + j, preds[j]))
                    one = targets[j].detach().cpu().numpy()
                    gts.append(one)

        m = {"val_loss": sums["loss"] / max(n_img, 1),
             "val_box": sums["box"] / max(n_img, 1),
             "val_obj": sums["obj"] / max(n_img, 1),
             "val_cls": sums["cls"] / max(n_img, 1)}
        m.update(self._mean_ap(records, gts, len(self._val_idx)))
        return m

    # ---- 6.7 终止判定（核心） ---------------------------------------------
    def should_stop(self, train_metrics: Any = None,
                    val_metrics: Any = None) -> bool:
        """判断要不要收工，并把原因写进 self.stop_reason。

        四条触发条件，按优先级：
            1) 训练损失出现 NaN / inf —— 立刻停，没有继续的意义
            2) 达到 target_metric     —— 提前成功收工
            3) 早停：提高率 < min_delta 且连续 patience 轮
            4) 轮数用尽
        """
        self.should_stop_flag = False
        vm = val_metrics or {}
        tm = train_metrics or {}

        # 1) 数值异常
        for key, val in (("train_loss", tm.get("loss")), ("val_loss", vm.get("val_loss"))):
            if val is None:
                continue
            if val != val or val in (float("inf"), float("-inf")):
                self.stop_reason = "第 %d 轮 %s 出现 NaN/inf，停机" % (self.epoch + 1, key)
                self.should_stop_flag = True
                return True

        # 2) 达到目标
        if self.target_metric is not None and self.stop_metric in vm:
            cur = vm[self.stop_metric]
            good = (cur <= self.target_metric if self._lower_better()
                    else cur >= self.target_metric)
            if good:
                self.stop_reason = ("第 %d 轮 %s=%.4f 达到目标 %.4f，提前收工"
                                    % (self.epoch + 1, self.stop_metric, cur,
                                       self.target_metric))
                self.should_stop_flag = True
                return True

        # 3) 早停：相对提高率
        if self.stop_metric not in vm:
            self.stop_reason = "验证集没有 %s 这个指标" % self.stop_metric
            self.should_stop_flag = True
            return True
        cur = float(vm[self.stop_metric])
        lower_better = self._lower_better()

        # 历史最优（只看数值方向，不管门槛）用于保存最优权重
        if self.best_metric is None or ((cur < self.best_metric) if lower_better
                                        else (cur > self.best_metric)):
            self.best_metric = cur
            self.best_epoch = self.epoch + 1
            self.save_checkpoint(best=True)

        # 另外单独存一份「mAP 最优」：早停看损失，部署看 mAP，两者会分家
        # （实测过：验证损失最低那一轮 mAP 0.365，mAP 最高那一轮损失高 13%）。
        mp = vm.get("mAP50")
        if mp is not None and mp == mp and (self.best_map is None or mp > self.best_map):
            self.best_map = float(mp)
            self.best_map_epoch = self.epoch + 1
            self.save_checkpoint(name="best_map.pt")

        if self._ref is None:
            self._ref = cur
            self._stale = 0
            return False

        denom = abs(self._ref) if abs(self._ref) > 1e-12 else 1e-12
        gain = (self._ref - cur) / denom if lower_better else (cur - self._ref) / denom

        if gain >= self.min_delta:
            self._ref = cur              # 有实质进步，把参考值往前推
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                self.stop_reason = (
                    "第 %d 轮早停：%s=%.5f，相对参考值 %.5f 的提高率 %.3f%% "
                    "低于门槛 %.3f%%，已连续 %d 轮（patience=%d）"
                    % (self.epoch + 1, self.stop_metric, cur, self._ref,
                       gain * 100, self.min_delta * 100, self._stale, self.patience))
                self.should_stop_flag = True
                return True
        return False

    def _lower_better(self) -> bool:
        """早停指标是越大越好还是越小越好。损失越小越好，mAP 越大越好。"""
        return self.stop_metric == "val_loss"

    # ---- 6.8 记录 ---------------------------------------------------------
    def log_epoch(self, train_metrics: Any, val_metrics: Any) -> None:
        """把这一轮的号码写进 history、终端和 CSV。

        CSV 是逐行追加的，训练中断也不会丢前面几轮的记录——这一点在
        「训练到第 80 轮崩了」的时候很有用。
        """
        lr = None
        if hasattr(self.optimizer, "current_lr"):
            lr = self.optimizer.current_lr()
        row = {
            "epoch": self.epoch + 1,
            "train_loss": train_metrics.get("loss"),
            "val_loss": val_metrics.get("val_loss"),
            "box": val_metrics.get("val_box"),
            "obj": val_metrics.get("val_obj"),
            "cls": val_metrics.get("val_cls"),
            "mAP50": val_metrics.get("mAP50"),
            "mAP50-95": val_metrics.get("mAP50-95"),
            "lr": lr,
            "grad_norm": train_metrics.get("grad_norm"),
            "seconds": val_metrics.get("seconds"),
        }
        self.history.append(row)
        print("[第 %3d 轮] 训练 %.4f | 验证 %.4f (box %.4f obj %.4f cls %.4f) "
              "| mAP50 %.4f | mAP50-95 %.4f | lr %.2e"
              % (row["epoch"], row["train_loss"] or 0, row["val_loss"] or 0,
                 row["box"] or 0, row["obj"] or 0, row["cls"] or 0,
                 row["mAP50"] or 0, row["mAP50-95"] or 0, row["lr"] or 0))
        if self._csv_path:
            with open(self._csv_path, "a", encoding="utf-8") as fh:
                fh.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s,%s\n" % (
                    row["epoch"], row["train_loss"] or 0, row["val_loss"] or 0,
                    row["box"] or 0, row["obj"] or 0, row["cls"] or 0,
                    row["mAP50"] or 0, row["mAP50-95"] or 0,
                    "%.8g" % row["lr"] if row["lr"] else "",
                    "%.4f" % row["grad_norm"] if row["grad_norm"] else "",
                    "%.1f" % row["seconds"] if row["seconds"] else ""))

    # ---- 6.9 存权重 -------------------------------------------------------
    def save_checkpoint(self, best: bool = False, name: Optional[str] = None) -> Any:
        """保存权重。best=True 存最优，否则存最近一轮（可续训）。

        用「先写临时文件再改名」的方式落盘：训练中途被打断时，磁盘上要么是
        上一份完整的权重，要么是新的一份，不会出现写了一半的坏文件。
        """
        import os as _os

        if not self.out_dir:
            return None
        torch, _ = self._import_torch()
        fname = name or ("best.pt" if best else "last.pt")
        path = _os.path.join(self.out_dir, "models", fname)
        tmp = path + ".tmp"
        payload = {
            "backbone": self._backbone.state_dict(),
            "head": self._head.state_dict(),
            "epoch": self.epoch + 1,
            "best_metric": self.best_metric,
            "stop_metric": self.stop_metric,
            "num_classes": getattr(self._head, "num_classes", None),
            "history": self.history,
        }
        opt = getattr(self.optimizer, "optimizer", self.optimizer)
        if hasattr(opt, "state_dict"):
            payload["optimizer"] = opt.state_dict()
        torch.save(payload, tmp)
        _os.replace(tmp, path)
        if best:
            print("  -> 保存最优权重（%s=%.5f）" % (self.stop_metric,
                                                  self.best_metric or 0))
        return path

    # ---- 6.10 过拟合 / 欠拟合判据 -----------------------------------------
    def detect_overfit(self, window: int = 5) -> Any:
        """拿 train / val 两条曲线做判断，给出证据而不是只给结论。

        判据（都用「连续 window 轮」的窗口，避免单轮抖动误判）：
            过拟合：训练损失连续下降 且 验证损失连续上升
            欠拟合：训练损失和验证损失都还在下降，且训练损失仍高于它的
                    历史最小值的 1.2 倍（说明还有得学）
            其余情况给「证据不足」或「正常」，不硬下结论。
        """
        h = [r for r in self.history if r.get("train_loss") is not None
             and r.get("val_loss") is not None]
        if len(h) < window + 1:
            return {"verdict": "证据不足",
                    "evidence": ["只有 %d 轮记录，至少需要 %d 轮"
                                 % (len(h), window + 1)]}

        tail = h[-(window + 1):]
        tr = [r["train_loss"] for r in tail]
        va = [r["val_loss"] for r in tail]
        tr_down = all(tr[i] > tr[i + 1] for i in range(window))
        va_up = all(va[i] < va[i + 1] for i in range(window))
        va_down = all(va[i] > va[i + 1] for i in range(window))

        evidence = ["最近 %d 轮：训练 %.4f -> %.4f，验证 %.4f -> %.4f"
                    % (window + 1, tr[0], tr[-1], va[0], va[-1])]
        ratio = va[-1] / tr[-1] if tr[-1] > 1e-12 else float("inf")
        evidence.append("验证/训练 损失比 %.2f" % ratio)

        if tr_down and va_up:
            return {"verdict": "过拟合", "evidence": evidence}
        best_tr = min(r["train_loss"] for r in h)
        if tr_down and va_down and tr[-1] > best_tr * 1.2:
            return {"verdict": "欠拟合", "evidence": evidence}
        return {"verdict": "正常", "evidence": evidence}

    # ---- 6.11 断点续训 ----------------------------------------------------
    def resume(self, ckpt: str) -> None:
        """从 checkpoint 续训：恢复权重、优化器状态、轮次与早停计数器。"""
        import os as _os

        torch, _ = self._import_torch()
        if not _os.path.exists(ckpt):
            raise FileNotFoundError("找不到 checkpoint：%s" % ckpt)
        if self._backbone is None:
            self.setup()
        blob = torch.load(ckpt, map_location=torch.device(self.device))
        self._backbone.load_state_dict(blob["backbone"])
        self._head.load_state_dict(blob["head"])
        if "optimizer" in blob:
            opt = getattr(self.optimizer, "optimizer", self.optimizer)
            if hasattr(opt, "load_state_dict"):
                opt.load_state_dict(blob["optimizer"])
        self.epoch = int(blob.get("epoch", 0))
        self.best_metric = blob.get("best_metric")
        self._ref = self.best_metric
        self._stale = 0
        print("[续训] 从第 %d 轮接着跑，当前最优 %s=%s"
              % (self.epoch, self.stop_metric, self.best_metric))

    # ---- 6.12 解码一批预测 ------------------------------------------------
    def _decode_batch(self, raws: Any) -> Any:
        """三层 raw -> 每张图一组 (框, 分数, 类别)，已做分数过滤与非极大值抑制。

        最终分数 = 类别分数 × 置信度：两个分支各管一件事，乘起来才是
        「这一格既有目标、又是这一类」的联合判断。只用一个会漏判。
        """
        torch, _ = self._import_torch()
        boxes, scores, objs, _ = self._head.decode(raws)
        conf, cls = scores.max(dim=-1)
        if objs is not None:
            conf = conf * objs.squeeze(-1)

        out = []
        for b in range(boxes.shape[0]):
            keep = conf[b] > self.conf_thres
            if int(keep.sum()) == 0:
                out.append((torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)))
                continue
            bx = boxes[b][keep]
            sc = conf[b][keep]
            cl = cls[b][keep]

            # 类内抑制：给不同类别的框加上不同的大偏移，让跨类框的 IoU 恒为 0，
            # 一次 NMS 就等价于「每个类别各抑制一遍」
            offset = cl.to(bx.dtype).unsqueeze(1) * 10000.0
            idx = self._nms(bx + offset, sc, self.nms_iou, self.max_det)
            out.append((bx[idx], sc[idx], cl[idx]))
        return out

    def _nms(self, boxes: Any, scores: Any, iou_thres: float,
             max_det: int) -> Any:
        """非极大值抑制：按分数从高到低，压掉与已保留框重叠过多的框。"""
        torch, _ = self._import_torch()
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0 and len(keep) < max_det:
            i = int(order[0])
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            iou = self._iou_one_to_many(boxes[i], boxes[rest])
            order = rest[iou <= iou_thres]
        return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)

    def _iou_one_to_many(self, one: Any, many: Any) -> Any:
        """一个框 [4] 与一组框 [M,4] 的 IoU（非极大值抑制内部用）。"""
        torch, _ = self._import_torch()
        if many.numel() == 0:
            return torch.zeros(0, device=one.device)
        x1 = torch.maximum(one[0], many[:, 0])
        y1 = torch.maximum(one[1], many[:, 1])
        x2 = torch.minimum(one[2], many[:, 2])
        y2 = torch.minimum(one[3], many[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        a1 = (one[2] - one[0]).clamp(min=0) * (one[3] - one[1]).clamp(min=0)
        a2 = (many[:, 2] - many[:, 0]).clamp(min=0) * (many[:, 3] - many[:, 1]).clamp(min=0)
        return inter / (a1 + a2 - inter).clamp(min=1e-12)

    # ---- 6.13 算 mAP ------------------------------------------------------
    def _mean_ap(self, records: Any, gts: Any, n_img: int) -> Any:
        """按 0.5:0.05:0.95 十个阈值算 mAP，返回 mAP50 / mAP50-95 / 逐类 AP50。

        「先配对再判类」：一个预测先要看它跟真值框重叠够不够（IoU 达标），
        才轮得到判类别对不对。顺序反了会把「框对类错」算成完全错误，
        而实际报告里这两件事要分开说。
        """
        import numpy as np

        nc = int(getattr(self._head, "num_classes", 6))
        # 把每张图的真值整理成 (框, 类别) 两个数组
        gt_by_img = []
        for one in gts:
            if one is None or len(one) == 0:
                gt_by_img.append((np.zeros((0, 4)), np.zeros(0, dtype=int)))
                continue
            cls = one[:, 0].astype(int)
            cx, cy, w, h = one[:, 1], one[:, 2], one[:, 3], one[:, 4]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            gt_by_img.append((boxes, cls))

        thresholds = np.arange(0.5, 1.0, 0.05)
        per_t = np.zeros(len(thresholds))
        ap50 = {}
        for ti, thr in enumerate(thresholds):
            aps = []
            for c in range(nc):
                ap = self._ap_class(records, gt_by_img, c, float(thr))
                if ap == ap:            # 不是 nan：这个类在验证集里有真值
                    aps.append(ap)
                    if ti == 0:
                        ap50[c] = float(ap)
            # 没有真值的类别不参与平均，否则一个小类缺样本就把整体拉成 nan
            per_t[ti] = float(np.mean(aps)) if aps else float("nan")
        return {"mAP50": float(per_t[0]),
                "mAP50-95": float(np.nanmean(per_t)),
                "ap50": ap50}

    @staticmethod
    def _ap_class(records: Any, gt_by_img: Any, cls_id: int,
                  thr: float) -> float:
        """单类别在某个 IoU 阈值下的平均精度（全点插值）。"""
        import numpy as np

        n_gt = sum(int((g[1] == cls_id).sum()) for g in gt_by_img)
        if n_gt == 0:
            return float("nan")

        # 收集这个类别的所有预测，按分数从高到低
        def _as_np(a: Any) -> Any:
            """张量和数组都接：训练里传张量，评估脚本里传 numpy。"""
            return a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)

        preds = []
        for img_i, (bx, sc, cl) in records:
            if len(sc) == 0:
                continue
            m = (_as_np(cl) == cls_id)
            if not m.any():
                continue
            b = _as_np(bx)[m]
            s = _as_np(sc)[m]
            for j in range(len(s)):
                preds.append((float(s[j]), img_i, b[j]))
        if not preds:
            return 0.0
        preds.sort(key=lambda t: -t[0])

        used = {i: np.zeros(len(g[0]), dtype=bool) for i, g in enumerate(gt_by_img)}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for k, (_, img_i, box) in enumerate(preds):
            gb, gc = gt_by_img[img_i]
            cand = np.where(gc == cls_id)[0]
            if cand.size == 0:
                fp[k] = 1
                continue
            iou = TrainingLoop._iou_np(box, gb[cand])
            prev_best = -1.0
            best = -1
            for j, iou_v in zip(cand, iou):
                if used[img_i][j] or iou_v < thr:
                    continue
                if iou_v > prev_best:
                    prev_best, best = iou_v, j
            if best >= 0:
                used[img_i][best] = True
                tp[k] = 1
            else:
                fp[k] = 1

        ct, cf = np.cumsum(tp), np.cumsum(fp)
        recall = ct / n_gt
        precision = ct / np.maximum(ct + cf, 1e-12)

        # 全点插值：把精度曲线从右往左改成单调不增，再按召回增量积分
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    @staticmethod
    def _iou_np(one: Any, many: Any) -> Any:
        """一个框与一组框的 IoU（numpy 版，供 mAP 用）。"""
        import numpy as np

        if many.shape[0] == 0:
            return np.zeros(0)
        x1 = np.maximum(one[0], many[:, 0])
        y1 = np.maximum(one[1], many[:, 1])
        x2 = np.minimum(one[2], many[:, 2])
        y2 = np.minimum(one[3], many[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = max((one[2] - one[0]) * (one[3] - one[1]), 0.0)
        a2 = np.clip(many[:, 2] - many[:, 0], 0, None) * np.clip(
            many[:, 3] - many[:, 1], 0, None)
        return inter / np.maximum(a1 + a2 - inter, 1e-12)


# ---------------------------------------------------------------------------
# 7. 最终输出类
# ---------------------------------------------------------------------------
class OutputExporter:
    """把训练产物落盘：权重、ONNX、曲线、日志、配置、一致性验证材料。

    题面 §6：导出 ONNX 用 onnxruntime CPU 推理；§6.3 必做 PyTorch 与 ONNX
    在同一批图上的 max|Δbox| / max|Δconf| 一致性验证；§9 交付物目录；
    §4 一键训练命令与依赖清单。
    待定：目录命名、onnx opset、动态/静态输入尺寸、曲线用 matplotlib 还是 PIL、
    manifest 里记哪些复现信息。
    """

    def __init__(self, config: Any = None, project_dir: Optional[str] = None,
                 model: Any = None, history: Any = None, classes: Any = None,
                 input_size: int = 640, opset: int = 17,
                 sample_images: Any = None, pad_value: int = 114) -> None:
        """装配导出所需的一切。除 project_dir 外都可以后配。

        参数:
            project_dir   项目根目录，产物写在它下面的 models/ 与 results/
            model         已经 assemble 好的完整模块（要能 model(x) 出三层 raw）
            history       TrainingLoop.history，画曲线用
            classes       {类别下标: 类别名}，缺省从 classes.txt 读
            opset         ONNX 算子集版本；17 是当前兼容面最广的一档
            sample_images 一致性验证用的图（numpy HWC uint8 列表），
                          不传就用固定种子的随机图，只验数值一致性
        """
        import os as _os

        if not isinstance(opset, int) or not 9 <= opset <= 20:
            raise ValueError("opset 要在 9~20 之间：%r" % (opset,))
        if not isinstance(input_size, int) or input_size < 32:
            raise ValueError("input_size 要是 >=32 的整数：%r" % (input_size,))

        self.config = config
        self.project_dir = project_dir or _os.path.dirname(_os.path.abspath(__file__))
        self.model_dir = _os.path.join(self.project_dir, "models")
        self.docs_dir = _os.path.join(self.project_dir, "docs")
        self.results_dir = _os.path.join(self.project_dir, "results")
        self.model = model
        self.history = history or []
        self.classes = classes or DataImporter._load_class_names()
        self.input_size = int(input_size)
        self.opset = int(opset)
        self.pad_value = int(pad_value)
        self.sample_images = sample_images
        self.onnx_path = _os.path.join(self.model_dir, "best.onnx")
        self.pt_path = _os.path.join(self.model_dir, "best.pt")
        self.preprocess_spec = {
            "size": self.input_size,
            "resize": "等比缩放 + 居中填充（letterbox）：短边缩到 %d，长边居中，"
                      "两边填常量" % self.input_size,
            "interp_down": "PIL BOX（面积平均，缩小用，等同于先低通再抽样）",
            "interp_up": "PIL BICUBIC（放大用）",
            "pad_value": self.pad_value,
            "normalize": "像素值除以 255；不减均值、不除标准差",
            "channel_order": "RGB；HWC -> CHW；batch 维在前",
            "input_shape": [1, 3, self.input_size, self.input_size],
            "dtype": "float32",
            "postprocess": "sigmoid 解码 -> 分数=类别分×置信度 -> 逐类 NMS "
                           "-> letterbox 反算回原图",
            "note": "训练端、ONNX 端、现场部署必须逐项一致；这是掉点的第一大原因",
        }
        self.manifest = {}

    # ---- 7.0 工具 ---------------------------------------------------------
    def _ensure_dirs(self) -> None:
        """建 models/ 与 results/，已存在就跳过。"""
        import os as _os
        for d in (self.model_dir, self.results_dir):
            _os.makedirs(d, exist_ok=True)

    @staticmethod
    def _sha256(path: str, chunk: int = 1 << 20) -> str:
        """算文件指纹，写进 manifest 供现场核对产物没被换过。"""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    @staticmethod
    def _write_json(path: str, obj: Any) -> str:
        """写 JSON，原子替换，避免中断留下半个文件。"""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path

    # ---- 7.1 存权重 -------------------------------------------------------
    def save_weights(self) -> Any:
        """把当前模型存到 models/best.pt。

        同时存下类别数、输入尺寸、类别映射，让这份权重自己能说明自己——
        部署端不用再去别处找配置。
        """
        import os as _os
        import torch

        self._ensure_dirs()
        if self.model is None:
            raise ValueError("self.model 是空的，先 ModelBuilder.assemble()")

        payload = {
            "model": self.model.state_dict(),
            "num_classes": len(self.classes),
            "input_size": self.input_size,
            "pad_value": self.pad_value,
            "classes": {str(k): v for k, v in self.classes.items()},
        }
        tmp = self.pt_path + ".tmp"
        torch.save(payload, tmp)
        _os.replace(tmp, self.pt_path)
        print("[导出] 权重 -> %s（%.1f MB）"
              % (self.pt_path, _os.path.getsize(self.pt_path) / 1048576))
        return self.pt_path

    # ---- 7.2 导出 ONNX ----------------------------------------------------
    def export_onnx(self) -> Any:
        """导出 models/best.onnx，并立刻用 onnxruntime 的 CPU 后端跑一次。

        四件事必须做对：
            * 输入尺寸固定（不做动态轴）。letterbox 已经把一切图都变成
              640x640，没有变尺寸需求；动态轴会让推理引擎放弃部分优化。
            * opset 17。太高老版本 onnxruntime 不认，太低缺算子。
            * 输出三个张量 p3/p4/p5，形状 [1,11,80,80] 等。**ONNX 里没有
              解码和 NMS**，那是部署端的事。
            * 导出后用 onnx.checker 验结构、用 ORT 真跑一次。导出不报错
              不代表能跑，这两步是分开的。
        """
        import os as _os
        import torch

        self._ensure_dirs()
        if self.model is None:
            raise ValueError("self.model 是空的，先 ModelBuilder.assemble()")

        model = self.model
        was = model.training
        model.eval()
        dummy = torch.zeros(1, 3, self.input_size, self.input_size)
        try:
            kwargs = dict(opset_version=self.opset,
                          input_names=["images"],
                          output_names=["p3", "p4", "p5"],
                          do_constant_folding=True)
            try:
                torch.onnx.export(model, dummy, self.onnx_path, dynamo=False,
                                  **kwargs)
            except TypeError:
                # 老一些的 torch 没有 dynamo 参数
                torch.onnx.export(model, dummy, self.onnx_path, **kwargs)
        finally:
            model.train(was)

        size_mb = _os.path.getsize(self.onnx_path) / 1048576.0
        print("[导出] ONNX -> %s（%.1f MB，opset %d）"
              % (self.onnx_path, size_mb, self.opset))

        # ---- 结构校验 ----
        import onnx
        onnx.checker.check_model(onnx.load(self.onnx_path))
        print("[导出] onnx.checker 结构校验通过")

        # ---- 用 CPU 后端真跑一次，确认算子都能落地 ----
        import onnxruntime as ort
        sess = ort.InferenceSession(self.onnx_path,
                                    providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"images": dummy.numpy()})
        shapes = [tuple(o.shape) for o in outs]
        print("[导出] onnxruntime CPUExecutionProvider 实跑通过：%s" % (shapes,))
        return {"path": self.onnx_path, "size_mb": round(size_mb, 2),
                "opset": self.opset, "outputs": shapes,
                "providers": sess.get_providers()}

    # ---- 7.3 画曲线 -------------------------------------------------------
    def save_loss_curve(self) -> Any:
        """把训练损失、验证指标、学习率画成三格图，存 results/curves.png。

        题面 §4 要求 loss 曲线与学习率策略留档。三格分开画而不是叠一张：
        损失在 1~30 之间、mAP 在 0~1 之间、学习率是 1e-5 量级，画在一起会
        把后两者压成一条直线。
        """
        import os as _os

        if not self.history:
            print("[导出] 没有 history，跳过曲线")
            return None
        self._ensure_dirs()

        import matplotlib
        matplotlib.use("Agg")           # 无显示环境必须用 Agg，否则报错
        import matplotlib.pyplot as plt

        ep = [r["epoch"] for r in self.history]
        tr = [r.get("train_loss") or 0 for r in self.history]
        va = [r.get("val_loss") or 0 for r in self.history]
        m50 = [r.get("mAP50") or 0 for r in self.history]
        m95 = [r.get("mAP50-95") or 0 for r in self.history]
        lr = [r.get("lr") or 0 for r in self.history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].plot(ep, tr, label="train")
        axes[0].plot(ep, va, label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend()
        axes[1].plot(ep, m50, label="mAP@0.5")
        axes[1].plot(ep, m95, label="mAP@0.5:0.95")
        axes[1].set_title("Validation mAP")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend()
        axes[2].plot(ep, lr, color="tab:orange")
        axes[2].set_title("Learning rate")
        axes[2].set_xlabel("epoch")
        fig.tight_layout()
        path = _os.path.join(self.results_dir, "curves.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("[导出] 曲线 -> %s" % path)
        return path

    # ---- 7.4 训练日志 -----------------------------------------------------
    def save_training_log(self) -> Any:
        """把逐轮记录落成 CSV + 一段文字摘要。

        CSV 是给人用 Excel 看的，摘要是给报告直接引用的：起始损失、最低
        验证损失出现在第几轮、最终 mAP。
        """
        import os as _os

        if not self.history:
            return None
        self._ensure_dirs()
        cols = ["epoch", "train_loss", "val_loss", "box", "obj", "cls",
                "mAP50", "mAP50-95", "lr", "grad_norm", "seconds"]
        path = _os.path.join(self.results_dir, "history.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(cols) + "\n")
            for r in self.history:
                fh.write(",".join("" if r.get(c) is None else str(r.get(c))
                                  for c in cols) + "\n")

        first, last = self.history[0], self.history[-1]
        best = min(self.history,
                   key=lambda r: r.get("val_loss") if r.get("val_loss") is not None
                   else float("inf"))
        lines = [
            "总轮数：%d" % len(self.history),
            "训练损失：%.4f -> %.4f" % (first.get("train_loss") or 0,
                                        last.get("train_loss") or 0),
            "验证损失：%.4f -> %.4f，最低 %.4f 出现在第 %d 轮"
            % (first.get("val_loss") or 0, last.get("val_loss") or 0,
               best.get("val_loss") or 0, best["epoch"]),
            "mAP@0.5：%.4f -> %.4f" % (first.get("mAP50") or 0,
                                       last.get("mAP50") or 0),
            "mAP@0.5:0.95：%.4f -> %.4f" % (first.get("mAP50-95") or 0,
                                            last.get("mAP50-95") or 0),
        ]
        txt = _os.path.join(self.results_dir, "history_summary.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("[导出] 训练日志 -> %s" % path)
        return path

    # ---- 7.5 配置 --------------------------------------------------------
    def save_config(self) -> Any:
        """把训练时的全部超参写成 results/train_config.json。

        内容以传入的 config 字典为准（训练循环自己填），这里补齐模型与数据
        两块的固定信息，保证「一键复现」时该有的都在一个文件里。
        """
        import os as _os

        self._ensure_dirs()
        cfg = dict(self.config) if isinstance(self.config, dict) else {
            "config": str(self.config)}
        cfg.setdefault("input_size", self.input_size)
        cfg.setdefault("pad_value", self.pad_value)
        cfg.setdefault("num_classes", len(self.classes))
        cfg.setdefault("classes", {str(k): v for k, v in self.classes.items()})
        cfg.setdefault("opset", self.opset)
        path = _os.path.join(self.results_dir, "train_config.json")
        self._write_json(path, cfg)
        print("[导出] 配置 -> %s" % path)
        return path

    # ---- 7.6 预处理规格（部署端共用） -------------------------------------
    def save_preprocess_spec(self) -> Any:
        """把预处理规格单独写成一份文件。

        为什么单独一份：部署端（deploy_cpu.py）和训练端必须做完全相同的事，
        而它们跑在两个不同的进程里、甚至两台不同的机器上。把规格写成文件，
        部署时读它、核对它，比「靠记忆保持一致」可靠。
        """
        import os as _os

        self._ensure_dirs()
        path = _os.path.join(self.results_dir, "preprocess_spec.json")
        self._write_json(path, self.preprocess_spec)
        print("[导出] 预处理规格 -> %s" % path)
        return path

    # ---- 7.7 一致性验证（§6.3 必做） --------------------------------------
    def verify_consistency(self) -> Any:
        """同一批图，PyTorch 与 ONNX 两条路径的输出差异。

        分两层比，因为意义不同：

            第一层 比原始输出张量。一一对应、没有排序，应该小到 1e-5 以下。
                   这一层大 = 导出或算子替换出了问题。
            第二层 比最终检测结果（框和分数）。这一层参与过 NMS，只要有一个
                   浮点差异就可能改变排序，所以**不能按下标硬比**，要按分数
                   排序后比、并且报告数量差。题面要的 max|Δbox| / max|Δconf|
                   就是这一层。

        超过 1e-3 要解释：最常见的原因是预处理不一致（第二常见是 NMS 细节）。
        """
        import os as _os
        import numpy as np
        import torch

        if self.model is None or not _os.path.exists(self.onnx_path):
            raise RuntimeError("先 save_weights / export_onnx，再做一致性验证")

        images = self.sample_images
        if not images:
            rng = np.random.default_rng(0)
            images = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
                      for _ in range(4)]

        det = Detector(onnx_path=self.onnx_path, input_size=self.input_size,
                       class_names=self.classes, backend="onnx")
        det.load()

        model = self.model
        was = model.training
        model.eval()

        raw_diff, box_diff, conf_diff, n_diff = 0.0, 0.0, 0.0, 0
        try:
            with torch.no_grad():
                for img in images:
                    blob, geom = det.preprocess(img)

                    out_t = model(torch.from_numpy(blob))
                    raws_t = [o.numpy() for o in out_t]

                    raws_o = det._run_onnx(blob)

                    for a, b in zip(raws_t, raws_o):
                        raw_diff = max(raw_diff, float(np.abs(a - b).max()))

                    d_t = det.postprocess(raws_t, geom)
                    d_o = det.postprocess(raws_o, geom)
                    n_diff = max(n_diff, abs(len(d_t) - len(d_o)))
                    m = min(len(d_t), len(d_o))
                    if m:
                        # 两边都已按分数从高到低排过，取前 m 个对齐再比
                        box_diff = max(box_diff,
                                       float(np.abs(d_t[:m, :4] - d_o[:m, :4]).max()))
                        conf_diff = max(conf_diff,
                                        float(np.abs(d_t[:m, 4] - d_o[:m, 4]).max()))
        finally:
            model.train(was)

        report = {
            "n_images": len(images),
            "max_abs_diff_raw": raw_diff,
            "max_abs_diff_box_px": box_diff,
            "max_abs_diff_conf": conf_diff,
            "max_count_diff": int(n_diff),
            "threshold": 1e-3,
            "pass": bool(raw_diff < 1e-3 and box_diff < 1e-3
                         and conf_diff < 1e-3 and n_diff == 0),
        }
        path = _os.path.join(self.results_dir, "consistency.json")
        self._write_json(path, report)
        print("[导出] 一致性：raw %.2e / box %.3e px / conf %.3e / 数量差 %d -> %s"
              % (raw_diff, box_diff, conf_diff, n_diff,
                 "通过" if report["pass"] else "未通过，需要解释"))
        return report

    # ---- 7.8 产物清单 -----------------------------------------------------
    def write_manifest(self) -> Any:
        """汇总所有产物 + 复现信息，写 results/manifest.json。

        记什么：每个文件的大小与 sha256（现场核对没被换过）、训练命令、
        依赖版本、类别映射、最终指标。这是「闭环完整性」这一项的证据。
        """
        import os as _os
        import platform

        self._ensure_dirs()
        files = {}
        for d in (self.model_dir, self.results_dir):
            if not _os.path.isdir(d):
                continue
            for name in sorted(_os.listdir(d)):
                p = _os.path.join(d, name)
                if _os.path.isfile(p):
                    files[_os.path.relpath(p, self.project_dir)] = {
                        "size": _os.path.getsize(p),
                        "sha256": self._sha256(p),
                    }

        import torch
        self.manifest = {
            "generated": __import__("datetime").datetime.now().isoformat(
                timespec="seconds"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "command": "python3 train.py --pictures data/annotations.json",
            "input_size": self.input_size,
            "num_classes": len(self.classes),
            "classes": {str(k): v for k, v in self.classes.items()},
            "epochs": len(self.history),
            "final": self.history[-1] if self.history else None,
            "files": files,
        }
        path = _os.path.join(self.results_dir, "manifest.json")
        self._write_json(path, self.manifest)
        print("[导出] 产物清单 -> %s（%d 个文件）" % (path, len(files)))
        return path


# ---------------------------------------------------------------------------
# 8. 检测类：拿训练好的模型去认图
# ---------------------------------------------------------------------------
class Detector:
    """用训练好的模型对图片 / 视频 / 摄像头做检测。

    两种后端，一个出口：

        onnx  跑 onnxruntime。现场部署用这个，不需要 torch、不需要训练代码。
        pt    直接跑 PyTorch。验证、对比、以及还没导出 ONNX 时用。

    三种输入共用同一条流水线，只写一次 ``detect``：

        一帧（RGB 数组）
          -> preprocess  等比缩放 + 居中填充 -> [1,3,S,S] float32
          -> 后端推理    三层 raw
          -> postprocess 解码 -> 分数=类别分×置信度 -> 逐类 NMS -> 反算回原图
          -> draw        框 + 类别名 + 置信度 + FPS

    为什么要分后端却共用一个后处理：ONNX 里没有解码和 NMS（它们含动态控制流，
    导出成图会变成一堆难调的算子，题面 §10 也禁止用现场可能没有的算子）。
    所以后处理必须在 Python 里做一份，两种后端都用这一份——这样「PyTorch 与
    ONNX 结果一致」这件事才只取决于导出本身，不取决于两份解码写得一不一样。
    """

    def __init__(self, weights: Optional[str] = None,
                 onnx_path: Optional[str] = None, num_classes: int = 6,
                 input_size: int = 640, conf_thres: float = 0.25,
                 nms_iou: float = 0.45, max_det: int = 100,
                 pad_value: int = 114, class_names: Any = None,
                 backend: str = "auto", device: str = "cpu",
                 font_path: Optional[str] = None) -> None:
        """参数都是「部署端要能独立配置」的东西，不依赖训练时的对象。

        backend：auto 时有 onnx 路径就走 onnx，否则走 pt。
        pad_value 必须与训练端一致（默认 114，YOLO 惯例的灰边）。
        """
        if backend not in ("auto", "onnx", "pt"):
            raise ValueError("backend 只支持 auto / onnx / pt：%r" % (backend,))
        if not 0 < conf_thres < 1:
            raise ValueError("conf_thres 要在 (0,1) 之间：%r" % (conf_thres,))
        if not 0 < nms_iou <= 1:
            raise ValueError("nms_iou 要在 (0,1] 之间：%r" % (nms_iou,))

        self.weights = weights
        self.onnx_path = onnx_path
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.conf_thres = float(conf_thres)
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)
        self.pad_value = int(pad_value)
        self.class_names = class_names or DataImporter._load_class_names()
        self.device = device
        self.font_path = font_path
        self.backend = backend if backend != "auto" else (
            "onnx" if onnx_path else "pt")
        self.session = None
        self.model = None
        self.input_name = None
        self.output_names = None
        self._font = None
        self.last_timing = {}           # {"pre","infer","post"} 秒
        self._hist = []                 # 最近几帧的检测结果，跨帧投票用

    # ---- 8.1 载入 ---------------------------------------------------------
    def load(self) -> Any:
        """按后端载入模型。ONNX 走 onnxruntime，pt 走 PyTorch。"""
        if self.backend == "onnx":
            if not self.onnx_path or not os.path.exists(self.onnx_path):
                raise FileNotFoundError("找不到 ONNX：%r" % (self.onnx_path,))
            import onnxruntime as ort
            # 现场只会用 CPU，屏蔽掉「找不到 GPU 设备」这类无关日志
            try:
                ort.set_default_logger_severity(3)
            except Exception:
                pass
            so = ort.SessionOptions()
            # 三个线程够用：现场是笔记本 CPU，线程开太多反而互相抢
            so.intra_op_num_threads = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.onnx_path, sess_options=so,
                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            print("[检测] 载入 ONNX：%s（%s）"
                  % (self.onnx_path, self.session.get_providers()[0]))
            return self

        if not self.weights or not os.path.exists(self.weights):
            raise FileNotFoundError("找不到权重：%r" % (self.weights,))
        import torch
        blob = torch.load(self.weights, map_location="cpu")
        sd = _merge_ckpt(blob)          # 兼容「分开存」和「整模型存」两种格式
        if "num_classes" in blob:
            self.num_classes = int(blob["num_classes"])
        if "input_size" in blob:
            self.input_size = int(blob["input_size"])
        if "classes" in blob and not self.class_names:
            self.class_names = {int(k): v for k, v in blob["classes"].items()}

        mb = ModelBuilder(num_classes=self.num_classes,
                          input_size=self.input_size)
        self.model = mb.assemble()
        self.model.load_state_dict(sd)
        self.model.eval()
        self.model.to(torch.device(self.device))
        print("[检测] 载入 PyTorch 权重：%s" % self.weights)
        return self

    # ---- 8.2 预处理（必须与训练端逐项一致） --------------------------------
    def preprocess(self, frame: Any, pad_value: Optional[int] = None) -> Any:
        """一帧 RGB 图 -> (blob [1,3,S,S] float32, 几何参数)。

        三件必须和训练端一样的事：
            * 等比缩放 + 居中填充，不是直接 resize。直接 resize 会把 1920x1080
              的图压扁，框整体变形，精度看起来「莫名掉了」。
            * 只除以 255，不减均值、不除标准差（训练端就是这么做的）。
            * RGB 顺序、HWC -> CHW、batch 维在前。
        """
        import numpy as np
        from PIL import Image

        pad = self.pad_value if pad_value is None else int(pad_value)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("帧要是 [H,W,3] 的 RGB 数组，收到 %r"
                             % (frame.shape,))
        h, w = frame.shape[:2]
        size = self.input_size
        r = min(size / float(w), size / float(h))
        nw, nh = int(round(w * r)), int(round(h * r))
        left, top = (size - nw) // 2, (size - nh) // 2

        res = getattr(Image, "Resampling", Image)
        interp = res.BOX if r < 1.0 else res.BICUBIC
        with Image.fromarray(frame) as im:
            small = np.asarray(im.resize((nw, nh), interp), dtype=np.uint8)
        canvas = np.full((size, size, 3), pad, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = small

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None].copy()
        return blob, {"scale": float(r), "pad_x": int(left), "pad_y": int(top),
                      "orig_hw": (int(h), int(w)), "size": int(size)}

    # ---- 8.3 推理 ---------------------------------------------------------
    def _run_onnx(self, blob: Any) -> Any:
        """onnxruntime 前向，返回三层 raw（按网格面积从大到小排序）。"""
        outs = self.session.run(self.output_names or None,
                                {self.input_name: blob})
        outs = list(outs)
        # 自研模型吐三层 [1,C,H,W]，按网格面积从大到小排，保证第 0 层最细；
        # YOLO 只吐一个 [1,4+类别,8400]，没有第 4 维，不能套这个排序。
        if len(outs) > 1 and all(o.ndim == 4 for o in outs):
            outs.sort(key=lambda a: a.shape[2] * a.shape[3], reverse=True)
        return outs

    def _run_torch(self, blob: Any) -> Any:
        """PyTorch 前向，返回三层 raw（numpy）。"""
        import torch
        with torch.no_grad():
            x = torch.from_numpy(blob).to(torch.device(self.device))
            raws = self.model(x)
        return [r.detach().cpu().numpy() for r in raws]

    def infer(self, blob: Any) -> Any:
        """按后端跑一次前向。"""
        if self.backend == "onnx":
            return self._run_onnx(blob)
        return self._run_torch(blob)

    # ---- 8.4 后处理（numpy 版，两种后端共用） -----------------------------
    def postprocess(self, raws: Any, geom: Any) -> Any:
        """三层 raw -> [n,6] 的检测结果（原图坐标）。

        列顺序：(左上横, 左上纵, 右下横, 右下纵, 置信度, 类别下标)，按分数降序。

        三步在做什么：
            解码   把「格点 + 偏移」还原成像素坐标。中心用 sigmoid 压在格内
                   （乘 2 减 0.5 后落在 -0.5~1.5 格），尺寸用 exp 保证为正。
            合成   分数 = 类别分 × 置信度。两个分支各管一件事，乘起来才是
                   「这一格有目标、且是这一类」的联合判断。
            抑制   逐类做非极大值抑制。用「给不同类加不同大偏移」的技巧：
                   加上偏移后跨类框的 IoU 恒为 0，一次抑制等价于每类各抑制一遍。
        """
        import numpy as np

        # ultralytics 导出的 YOLOv8 ONNX 只有一个输出，布局是 [1, 4+nc, 8400]，
        # 而且框已经解码到像素、类别分已经过 sigmoid。走单独一条分支，
        # 数学与 M6-1/infer.py 完全一致。
        if len(raws) == 1:
            return self._postprocess_yolo(np.asarray(raws[0], dtype=np.float32),
                                          geom)

        s_all, b_all, c_all = [], [], []
        for k, raw in enumerate(raws):
            raw = np.asarray(raw, dtype=np.float32)
            if raw.ndim == 4:
                raw = raw[0]
            ch, h, w = raw.shape
            s = float(self.input_size) / float(h) if h else 8.0
            # 网格：行 i、列 j 的整数下标
            gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

            t = raw[0:4]
            cx = (gx + 2.0 * self._sigmoid(t[0]) - 0.5) * s
            cy = (gy + 2.0 * self._sigmoid(t[1]) - 0.5) * s
            bw = np.exp(np.clip(t[2], -5.0, 5.0)) * s
            bh = np.exp(np.clip(t[3], -5.0, 5.0)) * s

            box = np.stack([cx - bw * 0.5, cy - bh * 0.5,
                            cx + bw * 0.5, cy + bh * 0.5], axis=-1).reshape(-1, 4)

            cls_p = self._sigmoid(raw[4:4 + self.num_classes]).reshape(
                self.num_classes, -1)
            conf = cls_p.max(axis=0)
            cls_id = cls_p.argmax(axis=0)
            if ch > 4 + self.num_classes:
                conf = conf * self._sigmoid(raw[4 + self.num_classes]).reshape(-1)

            s_all.append(conf)
            b_all.append(box)
            c_all.append(cls_id)

        conf = np.concatenate(s_all)
        box = np.concatenate(b_all).astype(np.float32)
        cls = np.concatenate(c_all).astype(np.int64)

        keep = conf > self.conf_thres
        if not keep.any():
            return np.zeros((0, 6), dtype=np.float32)
        box, conf, cls = box[keep], conf[keep], cls[keep]

        idx = self._nms_class_aware(box, conf, cls, self.nms_iou, self.max_det)
        box, conf, cls = box[idx], conf[idx], cls[idx]

        # ---- letterbox 反算回原图坐标 ----
        r = float(geom["scale"]) if geom.get("scale") else 1.0
        pad_x, pad_y = float(geom.get("pad_x", 0)), float(geom.get("pad_y", 0))
        oh, ow = geom.get("orig_hw", (self.input_size, self.input_size))
        box[:, 0] = np.clip((box[:, 0] - pad_x) / r, 0, ow)
        box[:, 1] = np.clip((box[:, 1] - pad_y) / r, 0, oh)
        box[:, 2] = np.clip((box[:, 2] - pad_x) / r, 0, ow)
        box[:, 3] = np.clip((box[:, 3] - pad_y) / r, 0, oh)

        out = np.concatenate([box, conf[:, None], cls[:, None].astype(np.float32)],
                             axis=1)
        order = np.argsort(-out[:, 4])
        return out[order]

    def _postprocess_yolo(self, out: Any, geom: Any) -> Any:
        """YOLOv8 单张量输出的解码：[1, 4+nc, 8400] -> [n,6]（原图坐标）。

        三处与自研模型不同的地方：
            * 输出是**一个**张量，8400 个预测点在最后一维，先转置再按行取。
            * 前 4 个数已经是 (cx,cy,w,h) 的**像素值**，不要再乘步长、不要 sigmoid。
            * 类别分**已经过 sigmoid**，直接取最大值；没有独立 objectness 通道。
        """
        import numpy as np

        pred = out[0] if out.ndim == 3 else out
        pred = np.squeeze(pred)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        conf = cls_scores.max(axis=1)
        cls = cls_scores.argmax(axis=1)

        keep = conf > self.conf_thres
        if not keep.any():
            return np.zeros((0, 6), dtype=np.float32)
        xywh, conf, cls = xywh[keep], conf[keep], cls[keep]

        cx, cy, w, h = xywh.T
        box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        idx = self._nms_class_aware(box, conf, cls, self.nms_iou, self.max_det)
        box, conf, cls = box[idx], conf[idx], cls[idx]

        r = float(geom["scale"]) if geom.get("scale") else 1.0
        pad_x, pad_y = float(geom.get("pad_x", 0)), float(geom.get("pad_y", 0))
        oh, ow = geom.get("orig_hw", (self.input_size, self.input_size))
        box[:, 0] = np.clip((box[:, 0] - pad_x) / r, 0, ow)
        box[:, 1] = np.clip((box[:, 1] - pad_y) / r, 0, oh)
        box[:, 2] = np.clip((box[:, 2] - pad_x) / r, 0, ow)
        box[:, 3] = np.clip((box[:, 3] - pad_y) / r, 0, oh)

        res = np.concatenate([box, conf[:, None], cls[:, None].astype(np.float32)],
                             axis=1)
        return res[np.argsort(-res[:, 4])]

    @staticmethod
    def _sigmoid(x: Any) -> Any:
        """1/(1+e^-x)，数值稳定写法（正负分段，避免 exp 溢出）。"""
        import numpy as np
        out = np.empty_like(x, dtype=np.float32)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    @staticmethod
    def _iou_nm(a: Any, many: Any) -> Any:
        """一个框与一组框的 IoU（numpy）。"""
        import numpy as np
        x1 = np.maximum(a[0], many[:, 0])
        y1 = np.maximum(a[1], many[:, 1])
        x2 = np.minimum(a[2], many[:, 2])
        y2 = np.minimum(a[3], many[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = max((a[2] - a[0]) * (a[3] - a[1]), 0.0)
        a2 = np.clip(many[:, 2] - many[:, 0], 0, None) * np.clip(
            many[:, 3] - many[:, 1], 0, None)
        return inter / np.maximum(a1 + a2 - inter, 1e-12)

    def _nms_class_aware(self, box: Any, conf: Any, cls: Any,
                         iou_thres: float, max_det: int) -> Any:
        """逐类非极大值抑制，返回保留框的下标（按分数降序）。

        技巧：给不同类别的框加上不同的巨大偏移，跨类框的重叠度就成了 0，
        于是一次抑制等价于「每类各抑制一遍」，不用写循环。
        """
        import numpy as np

        order = np.argsort(-conf)
        shifted = box[order] + cls[order][:, None].astype(np.float32) * 10000.0
        keep = []
        alive = np.arange(len(order))
        while alive.size and len(keep) < max_det:
            i = alive[0]
            keep.append(int(order[i]))
            if alive.size == 1:
                break
            rest = alive[1:]
            iou = self._iou_nm(shifted[i], shifted[rest])
            alive = rest[iou <= iou_thres]
        return np.asarray(keep, dtype=np.int64)

    # ---- 8.5 一帧到底 -----------------------------------------------------
    def smooth(self, dets: Any, window: int = 3, need: int = 2) -> Any:
        """跨帧投票：只保留「最近 window 帧里至少出现 need 次」的框。

        为什么需要：摄像头逐帧的检测结果本来就抖 —— 一个置信度 0.26 的假阳性
        下一帧变成 0.24 就消失了，屏幕上就是「框一闪一闪」。真实物体会连续
        出现在每一帧，假阳性不会。所以「连续出现几次」是一个几乎零成本的
        过滤器，比调高阈值更有效（调高阈值会把真目标一起杀掉）。

        匹配条件：类别相同 + 框重叠度 > 0.4。用重叠度而不是下标，因为每帧的
        框数量和顺序都会变。
        """
        import numpy as np

        self._hist.append(dets)
        if len(self._hist) > window:
            self._hist.pop(0)
        if len(dets) == 0:
            return dets

        keep = []
        for one in dets:
            support = 0
            for past in self._hist:
                if len(past) == 0:
                    continue
                iou = self._iou_nm(one[:4], past[:, :4])
                same = (past[:, 5].astype(int) == int(one[5]))
                if np.any((iou > 0.4) & same):
                    support += 1
            if support >= need:
                keep.append(one)
        if not keep:
            return np.zeros((0, 6), dtype=np.float32)
        return np.stack(keep).astype(np.float32)

    def detect(self, frame: Any) -> Any:
        """一帧 RGB 图 -> 检测结果，顺便把三段耗时记进 self.last_timing。"""
        import time

        t0 = time.perf_counter()
        blob, geom = self.preprocess(frame)
        t1 = time.perf_counter()
        raws = self.infer(blob)
        t2 = time.perf_counter()
        dets = self.postprocess(raws, geom)
        t3 = time.perf_counter()
        self.last_timing = {"pre": t1 - t0, "infer": t2 - t1, "post": t3 - t2}
        return dets

    # ---- 8.6 画框（中文要用 PIL，cv2 画不了） -----------------------------
    def _get_font(self) -> Any:
        """找一个能显示中文的字体。找不到就退回 ASCII 标注。

        为什么必须特别处理：OpenCV 的 cv2.putText 只支持 ASCII，直接画
        「保温杯」会变成一串问号。这是现场演示最容易翻车的地方，所以这里
        用 PIL 画字，并准备了一条退路（找不到字体就画 cls 编号）。
        """
        if self._font is not None:
            return self._font
        from PIL import ImageFont

        cands = [self.font_path] if self.font_path else []
        cands += [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/mnt/c/Windows/Fonts/msyh.ttc",
            "/mnt/c/Windows/Fonts/simhei.ttf",
            "/mnt/c/Windows/Fonts/simsun.ttc",
        ]
        for p in cands:
            if p and os.path.exists(p):
                try:
                    self._font = ImageFont.truetype(p, 20)
                    self.font_path = p
                    return self._font
                except OSError:
                    continue
        self._font = False              # 记下「没有可用字体」，下次不再找
        return None

    def draw(self, frame: Any, dets: Any, fps: Optional[float] = None,
             ms: Optional[float] = None, note: Optional[str] = None) -> Any:
        """把框、类别名、置信度、FPS 画到图上，返回新图（RGB）。

        用 PIL 而不是 cv2：一是中文，二是线宽和字体都能控制。
        """
        import numpy as np
        from PIL import Image, ImageDraw

        img = Image.fromarray(np.ascontiguousarray(frame))
        draw = ImageDraw.Draw(img)
        font = self._get_font()

        palette = [(220, 60, 60), (60, 140, 220), (60, 170, 110),
                   (230, 170, 40), (160, 90, 200), (200, 90, 140)]

        for one in dets:
            x1, y1, x2, y2, score, cid = (float(v) for v in one)
            c = int(cid)
            color = palette[c % len(palette)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            name = self.class_names.get(c, str(c))
            label = "%s %.2f" % (name, score)
            if font is None:                 # 没有中文字体时退回 ASCII
                label = "c%d %.2f" % (c, score)
            tw, th = (draw.textbbox((0, 0), label, font=font)[2:4]
                      if font else (len(label) * 8, 16))
            ty = max(y1 - th - 4, 0)
            draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
            draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)

        head = []
        if fps is not None:
            head.append("FPS %.1f" % fps)
        if ms is not None:
            head.append("%.1f ms" % ms)
        if head:
            text = "  ".join(head)
            draw.rectangle([8, 8, 8 + len(text) * 11, 34], fill=(0, 0, 0))
            draw.text((12, 11), text, fill=(0, 255, 90), font=font)
        if note:
            draw.rectangle([8, 40, 8 + len(note) * 13, 68], fill=(150, 20, 20))
            draw.text((12, 44), note, fill=(255, 255, 255), font=font)
        return np.array(img)

    # ---- 8.7 三种输入 -----------------------------------------------------
    def run_image(self, src: str, out_path: Optional[str] = None,
                  show: bool = False) -> Any:
        """图片模式：读一张、跑一次、存一张带框图。"""
        import numpy as np
        from PIL import Image

        with Image.open(src) as im:
            frame = np.array(im.convert("RGB"))
        dets = self.detect(frame)
        vis = self.draw(frame, dets)
        if out_path:
            Image.fromarray(vis).save(out_path)
        if show:
            import cv2
            cv2.imshow("detect", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for one in dets:
            print("  %-6s %.3f  [%.0f, %.0f, %.0f, %.0f]"
                  % (self.class_names.get(int(one[5]), int(one[5])), one[4],
                     one[0], one[1], one[2], one[3]))
        return vis, dets

    def run_video(self, src: str, out_path: Optional[str] = None,
                  show: bool = False, max_frames: Optional[int] = None) -> Any:
        """视频模式：逐帧读、逐帧检测、可选写回 mp4。

        视频文件和检测是解耦的：第 N 帧进去、结果第 N 帧出来，检测多慢都
        不影响输出视频的时长（输出仍是原帧率，只是处理得慢）。
        """
        import time

        import cv2

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError("打不开视频：%s" % src)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if out_path:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_in, (w, h))

        n, t0 = 0, time.perf_counter()
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                dets = self.detect(frame)
                n += 1
                fps = n / max(time.perf_counter() - t0, 1e-6)
                vis = self.draw(frame, dets, fps=fps,
                                ms=sum(self.last_timing.values()) * 1000)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                if writer is not None:
                    writer.write(vis_bgr)
                if show:
                    cv2.imshow("detect", vis_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if max_frames and n >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        print("[视频] 处理 %d 帧，平均 %.1f FPS" % (n, n / max(
            time.perf_counter() - t0, 1e-6)))
        return n

    def _looks_green(self, bgr: Any) -> bool:
        """判断这一帧是不是「绿屏」。

        成因：摄像头默认用 YUYV（未压缩的 YUV）输出，而 WSL 的 UVC 驱动把
        它当 BGR 交给 OpenCV，色度信息整个错位，画面就成了纯绿。
        判据很直接：绿通道远高于红蓝两通道，且整体亮度不低。
        """
        import numpy as np
        if bgr is None or bgr.ndim != 3:
            return False
        b = float(np.mean(bgr[:, :, 0]))
        g = float(np.mean(bgr[:, :, 1]))
        r = float(np.mean(bgr[:, :, 2]))
        return g > 60.0 and g > 1.6 * max(b, r, 1.0)

    def _open_camera(self, index: int, size: Any = (640, 480),
                     fourcc: str = "MJPG") -> Any:
        """打开摄像头并挑一个能出正常颜色的采集格式。

        依次尝试 MJPG -> YUYV -> 不指定，哪个读出来的帧不是绿屏就用哪个。
        实测这台机器：默认走 YUYV 会绿屏，强制 MJPG 后颜色正常。
        分辨率也不宜往上顶：原生是 640x480，要 1280x720 时下半幅是黑的
        （传感器没那么大，驱动补了黑边），所以默认就用 640x480，
        想看清楚靠显示窗口放大，不靠提高采集分辨率。
        """
        import cv2

        tried = []
        for cc in (fourcc, "YUYV", ""):
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            if cc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            frame = None
            for _ in range(8):          # 丢几帧，等格式真正生效
                ok, frame = cap.read()
                if not ok:
                    frame = None
            if frame is None:
                tried.append("%s:读不到帧" % (cc or "默认"))
                cap.release()
                continue
            if self._looks_green(frame):
                tried.append("%s:绿屏" % (cc or "默认"))
                cap.release()
                continue
            real = "".join([chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> (8 * i)) & 0xFF)
                            for i in range(4)])
            print("[摄像头] 格式 %s  %dx%d  %s"
                  % (real, frame.shape[1], frame.shape[0],
                     ("（跳过：" + "、".join(tried) + "）") if tried else ""))
            return cap
        raise RuntimeError(
            "摄像头 %d 打开失败或一直绿屏（试过：%s）。\n"
            "  1) 换一个序号：--source 1 / 2 / 3\n"
            "  2) 关掉其它占用摄像头的程序（腾讯会议、相机 App 等）\n"
            "  3) 绿屏是 YUYV 解码问题，本机强制 MJPG 可解，若仍绿屏说明\n"
            "     WSL 的 USB 穿透没把这个摄像头完整映射过来"
            % (index, "、".join(tried) or "无"))

    def run_camera(self, index: int = 0, show: bool = True,
                   max_frames: Optional[int] = None,
                   size: Any = (640, 480), fourcc: str = "MJPG",
                   window_scale: float = 1.6, exposure: int = -1,
                   smooth: int = 3) -> Any:
        """摄像头模式：边读边检测边显示，按 q 退出。

        这一条才是现场演示的那条。帧率 = 1 /（读帧 + 预处理 + 推理 + 后处理
        + 画框 + 显示），任何一环拖慢都会直接压低 FPS。题面要求 CPU 上 ≥10 FPS。

        参数:
            exposure  >=0 时锁定曝光（0.25 手动模式），-1 表示自动。
                      自动曝光在「大面积暗 + 一个亮点」的场景里会来回搜索，
                      画面一明一暗地闪；锁死就稳了。
            smooth    跨帧投票的窗口，>=2 生效。摄像头逐帧结果本来会抖，
                      假阳性一帧有一帧没，看起来就是框在闪；连续出现够
                      次数才画，能把这个抖压掉。
        """
        import time

        import cv2

        cap = self._open_camera(index, size=size, fourcc=fourcc)
        self._hist = []                     # 每轮开始清空投票历史
        # 曝光是设备级状态：V4L2 把控制值存在驱动里，进程退出不会恢复。
        # 所以「不设置」不等于「自动」——之前任何一次手动锁定都会一直留着，
        # 表现就是「怎么都这么暗」。必须每次都显式设一次。
        #
        # 坑：OpenCV 文档说 0.25=手动、0.75=自动，但这个版本是直接透传
        # V4L2 原始枚举的：1 = 手动，3 = 自动（光圈优先）。传 0.75 会被
        # 截成 1，也就是悄悄锁成手动——这正是「画面一直很暗」的由来。
        if exposure is None or exposure < 0:
            ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
            for _ in range(12):
                cap.read()
            print("[摄像头] 曝光自动%s" % ("" if ok else "（设置失败，可能仍是手动）"))
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
            for _ in range(8):
                cap.read()
            print("[摄像头] 曝光锁定 %d，白平衡锁定" % exposure)

        win = "M2-3 detect (q = quit)"
        if show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, int(size[0] * window_scale),
                             int(size[1] * window_scale))
        n, t0, fps = 0, time.perf_counter(), 0.0
        dark_n = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                dets = self.detect(frame)
                if smooth and smooth >= 2:
                    dets = self.smooth(dets, window=int(smooth),
                                       need=max(2, int(smooth) - 1))
                n += 1
                if n % 5 == 0 or fps == 0:
                    fps = n / max(time.perf_counter() - t0, 1e-6)

                mean = float(frame.mean())
                note = None
                if mean < 25.0:
                    dark_n += 1
                    note = "画面过暗 %.0f/255，请开灯" % mean
                vis = self.draw(frame, dets, fps=fps,
                                ms=sum(self.last_timing.values()) * 1000,
                                note=note)
                if show:
                    cv2.imshow(win, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if max_frames and n >= max_frames:
                    break
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
        if dark_n and dark_n > n * 0.6:
            print("[提示] %d/%d 帧画面过暗（均值 < 25/255）。这种亮度下模型"
                  "必然检不出东西，请开灯或把物料挪到光源下。" % (dark_n, n))
        print("[摄像头] 处理 %d 帧，平均 %.1f FPS" % (n, n / max(
            time.perf_counter() - t0, 1e-6)))
        return n


# ---------------------------------------------------------------------------
# 入口：一键训练 + 导出
# ---------------------------------------------------------------------------
def _merge_ckpt(blob: Any) -> Any:
    """把两种权重格式统一成「一个 state_dict」。

    训练循环存的是 {"backbone": {...}, "head": {...}}（两块分开，方便续训时
    分别处理）；导出与部署要的是拼好的整模型。这里做一次合并，前缀分别是
    "backbone." 和 "head."，正好对上 ModelBuilder.assemble() 建出来的结构。
    """
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"]
    if isinstance(blob, dict) and "backbone" in blob and "head" in blob:
        merged = {}
        for k, v in blob["backbone"].items():
            merged["backbone." + k] = v
        for k, v in blob["head"].items():
            merged["head." + k] = v
        return merged
    return blob        # 已经就是一个裸 state_dict
def main() -> int:
    """一键训练：读标注 → 增强 + letterbox → 划分 → 建模 → 训练 → 导出。

    命令行::

        python3 train.py                        默认读同目录 annotations.json
        python3 train.py --epochs 20 --batch 8  快速验证用
        python3 train.py --no-train             只准备数据、看画像
        python3 train.py --no-export            只训练，不导 ONNX

    每一步都打印足够的信息，中断之后能一眼看出停在哪一环。
    """
    started = __import__("time").time()
    try:
        args = DataImporter().parse_args()
        importer = DataImporter(input_size=args.size, seed=args.seed,
                                augment=not args.no_augment)

        X, Y = importer.load_annotations(args.pictures)
        print("读到 %d 张原始样本、%d 个框（来自 %s）"
              % (len(X), sum(len(y) for y in Y), args.pictures))

        # 逐张增强 + letterbox，处理完一张就释放一张原图，峰值只保留一份原图。
        # 这一步会把 X 里用过的原图置成 None，所以之后不要再读 X 的原图。
        X, Y = importer.build_dataset(X, Y)
        importer.summary(X, Y)
        if args.no_train:
            print("--no-train：数据准备完就退出")
            return 0

        # 场景映射：把 data/splits/scene_map.json 放好就自动生效，
        # 形如 {"IMG_0001.jpg": "桌面A-自然光"}；没有就按「每张原图一个场景」划分。
        smap = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "splits", "scene_map.json")
        splitter = DatasetSplitter(
            seed=args.seed, scene_map=smap if os.path.isfile(smap) else None)
        splits = splitter.split(X, Y)
        splitter.balance_report(X, Y)
        splitter.write_splits(X)
        print("划分：train %d / val %d / test %d"
              % (len(splits["train"]), len(splits["val"]), len(splits["test"])))

        # ---- 建模 ----
        builder = ModelBuilder(num_classes=6, input_size=args.size,
                               device="auto")
        builder.build_backbone()
        builder.build_head()
        builder.assemble()
        print(builder.summary())
        print("前向形状检查：%s" % (builder.forward_shape_check(),))

        # ---- 损失、优化器、训练 ----
        loss_fn = LossFunction(num_classes=6, head=builder.head)
        # 预热轮数不能超过总轮数的一半：短跑（比如 --epochs 2）自动缩短，
        # 否则学习率还没升到目标值就已经结束了
        warmup = float(min(3, max(0, args.epochs // 4)))
        optim = OptimizerBuilder(lr=args.lr, batch_size=args.batch,
                                 total_epochs=args.epochs,
                                 warmup_epochs=min(warmup, args.epochs - 1))
        loop = TrainingLoop(max_epochs=args.epochs, batch_size=args.batch,
                            device="auto")
        loop.model = builder
        loop.loss_fn = loss_fn
        loop.optimizer = optim
        loop.datasets = {"X": X, "Y": Y, "splits": splits}
        loop.setup()

        try:
            loop.run()
        except KeyboardInterrupt:
            # 手动中断也要把已有权重导出，别让几十分钟白跑
            print("\n收到中断：用当前轮次的权重继续后面的导出")

        print("过拟合/欠拟合判据：%s" % (loop.detect_overfit(),))
        if args.no_export:
            print("--no-export：训练完就退出")
            return 0

        # ---- 导出（导出要求模型在 CPU 上，现场无 GPU）----
        # 优先载回 mAP 最优的权重：训练最后一轮通常已经过拟合，而早停用的是
        # 验证损失、部署看的是 mAP，两者会分家（best_map.pt 就是为这个存的）。
        for cand in ("best_map.pt", "best.pt"):
            best_ckpt = os.path.join(loop.out_dir, "models", cand)
            if os.path.exists(best_ckpt):
                import torch as _torch
                blob = _torch.load(best_ckpt, map_location="cpu")
                builder.model.load_state_dict(_merge_ckpt(blob))
                print("[导出] 已载回 %s：第 %s 轮" % (cand, blob.get("epoch")))
                break
        builder.model.cpu()
        builder.model.eval()
        exporter = OutputExporter(
            config={
                "epochs": args.epochs, "batch_size": args.batch, "lr": args.lr,
                "optimizer": optim.state_summary(),
                "loss": {"box_gain": 7.5, "obj_gain": 1.0, "cls_gain": 0.5,
                         "ignore_iou": loss_fn.ignore_iou,
                         "neighbor": loss_fn.neighbor},
                "augment": not args.no_augment, "seed": args.seed,
                "split_ratio": [0.7, 0.2, 0.1],
                "train_images": len(splits["train"]),
                "val_images": len(splits["val"]),
                "test_images": len(splits["test"]),
            },
            model=builder.model, history=loop.history, input_size=args.size)
        exporter.save_weights()
        exporter.export_onnx()
        exporter.save_loss_curve()
        exporter.save_training_log()
        exporter.save_config()
        exporter.save_preprocess_spec()
        exporter.verify_consistency()
        exporter.write_manifest()

        print("全部完成，用时 %.1f 分钟" % ((__import__("time").time() - started)
                                            / 60.0))
        return 0
    except MemoryError:
        print("内存不足：先把 --size 调小（比如 512），或者加 --no-augment "
              "只跑 resize。")
        return 1
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        print("训练流程失败：%s: %s" % (type(exc).__name__, exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
