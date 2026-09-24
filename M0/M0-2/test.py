import sys

sys.path.insert(0, "/home/xiaolingxiaotan/智能车一轮考核/智能车一轮考核/M0/M0-2")
from legacy_corr import sum, mean, r_calc

def run_test_case(xs, ys, expect_r=None, expect_exception=None):
    n = len(xs)
    try:
        if n <= 0:
            raise ValueError("无有效数据")
        r, mx, my = r_calc(xs, ys, n)
        print(f"计算得到 r = {r}")
        if expect_r is not None:
            if abs(r - expect_r) < 1e-6:
                print("测试通过")
            else:
                print(f"测试失败，预期 r = {expect_r}")
            if expect_exception is not None:
                print("测试失败：预期抛出异常，但没有异常")
    except Exception as e:
        print(f"捕获异常: {type(e).__name__}: {e}")
        if expect_exception is not None and isinstance(e, expect_exception):
            print("异常测试通过")
        else:
            if expect_exception is not None:
                print(f"异常类型不匹配，预期 {expect_exception.__name__}")
            else:
                print(f"意外捕获异常，本用例不应报错")
def main():
    # 用例1：完全正相关 r=1.0
    run_test_case(xs=[1,2,3], ys=[2,4,6], expect_r=1.0)

    # 用例2：完全负相关 r=-1.0
    run_test_case(xs=[1,2,3], ys=[6,4,2], expect_r=-1.0)

    # 用例3：错误测试
    run_test_case(xs=[5,5,5], ys=[1,2,3], expect_exception=ZeroDivisionError)

    run_test_case(xs=[], ys=[], expect_exception=ValueError)

if __name__ == "__main__":
    main()