"""
核心计算正确性测试（纯函数，不碰文件 IO）。

运行：
    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

# 让测试能 import 到上一级的 legacy_corr。
# 用「本文件所在位置」推导项目根目录，不写死绝对路径，换机器也能跑。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from legacy_corr import mean, r_calc
from legacy_corr import sum as sum_xy


class TestSumAndMean(unittest.TestCase):
    """基础统计量：求和与均值"""

    def test_sum(self):
        sx, sy = sum_xy([1, 2, 3], [2, 4, 6], 3)
        self.assertAlmostEqual(sx, 6.0, places=12)
        self.assertAlmostEqual(sy, 12.0, places=12)

    def test_mean(self):
        mx, my = mean([1, 2, 3], [2, 4, 6], 3)
        self.assertAlmostEqual(mx, 2.0, places=12)
        self.assertAlmostEqual(my, 4.0, places=12)


class TestRCalcKnownValues(unittest.TestCase):
    """题面点名的两个已知值用例"""

    def test_perfect_positive_correlation(self):
        self.assertAlmostEqual(r_calc([1, 2, 3], [2, 4, 6], 3), 1.0, places=12)

    def test_perfect_negative_correlation(self):
        self.assertAlmostEqual(r_calc([1, 2, 3], [6, 4, 2], 3), -1.0, places=12)


class TestRCalcAgainstNumpy(unittest.TestCase):
    """
    期望值不手算，直接让 numpy.corrcoef 现算一个独立结果来比。
    手写的期望值一旦算错，测试会跟着一起错；换一个独立实现做参照才有意义。
    """

    def assertMatchesNumpy(self, xs, ys):
        expected = float(np.corrcoef(np.asarray(xs, float), np.asarray(ys, float))[0, 1])
        self.assertAlmostEqual(r_calc(xs, ys, len(xs)), expected, places=12)

    def test_zero_correlation(self):
        # xs 与 ys 均值化后正交，相关系数恰为 0
        self.assertMatchesNumpy([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 1.0, 2.0])

    def test_mid_range(self):
        self.assertMatchesNumpy(
            [12.0, 15.5, 9.25, 20.0, 13.75],
            [3.0, 7.5, 2.0, 9.0, 5.25],
        )

    def test_negative_trend(self):
        self.assertMatchesNumpy(
            [1.0, 2.0, 3.0, 5.0],
            [-2.0, -4.0, -7.0, -9.0],
        )


class TestRCalcProperties(unittest.TestCase):
    """
    不依赖任何具体期望值的性质测试。
    相关系数对 x、y 的仿射变换（平移 + 正比例缩放）不变，对调 x/y 也不变。
    公式里少写一个负号、错乘一个符号，这类性质往往比单个数值更容易抓到。
    """

    XS = [1.0, 2.0, 4.0, 5.0, 8.0]
    YS = [2.0, 3.0, 3.5, 7.0, 6.5]

    def test_swap_x_y(self):
        self.assertAlmostEqual(
            r_calc(self.XS, self.YS, 5), r_calc(self.YS, self.XS, 5), places=12
        )

    def test_shift_and_positive_scale_x(self):
        scaled = [3.0 * v + 7.0 for v in self.XS]
        self.assertAlmostEqual(
            r_calc(self.XS, self.YS, 5), r_calc(scaled, self.YS, 5), places=12
        )

    def test_shift_and_positive_scale_y(self):
        scaled = [0.5 * v - 100.0 for v in self.YS]
        self.assertAlmostEqual(
            r_calc(self.XS, self.YS, 5), r_calc(self.XS, scaled, 5), places=12
        )

    def test_value_stays_within_range(self):
        r = r_calc(self.XS, self.YS, 5)
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)


class TestRCalcDegenerateInput(unittest.TestCase):
    """
    方差为 0（某一列取值恒定）时相关系数无定义，必须抛 ZeroDivisionError。
    这是入口 main 捕获它并转成退出码 3 的前提，所以在这里钉住。
    """

    def test_constant_x(self):
        with self.assertRaises(ZeroDivisionError):
            r_calc([5.0, 5.0, 5.0], [1.0, 2.0, 3.0], 3)

    def test_constant_y(self):
        with self.assertRaises(ZeroDivisionError):
            r_calc([1.0, 2.0, 3.0], [4.0, 4.0, 4.0], 3)

    def test_single_sample(self):
        # 只有一个样本时两列方差都是 0，同样无定义
        with self.assertRaises(ZeroDivisionError):
            r_calc([1.0], [2.0], 1)


if __name__ == "__main__":
    unittest.main()
