"""
CLI 契约与异常输入测试：以子进程方式运行 legacy_corr.py，检查退出码和输出。

为什么必须跑子进程，而不是在测试进程里 import 后直接调：
异常分支是靠 sys.exit(码) 结束的，SystemExit 在测试进程内一触发就会把整个
测试跑带停；而且题面要求"stdout 必须含 r = <数值>"、"不得把栈回溯抛给用户"，
这两条只有在真实进程的 stdout / stderr 上才验得了。

运行：
    python3 -m unittest discover -s tests -v
"""

import os
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(PROJECT_ROOT, "legacy_corr.py")

# 1,2 / 2,4 / 3,6 三行数据完全正相关，相关系数应为 1.0
GOOD_CSV = (
    "timestamp,sensor_a,sensor_b,note\n"
    "1,1,2,a\n"
    "2,2,4,b\n"
    "3,3,6,c\n"
)


class CliTestCase(unittest.TestCase):
    """提供一个临时工作目录，在其中写 yaml/csv 并以子进程运行入口脚本"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def write_csv(self, text, name="data.csv"):
        return self.write(name, text)

    def write_yaml(self, csv_name, col_x="sensor_a", col_y="sensor_b", name="config.yaml"):
        return self.write(
            name,
            'input_csv: "%s"\n'
            "columns:\n"
            '  x: "%s"\n'
            '  y: "%s"\n' % (csv_name, col_x, col_y),
        )

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, ENTRY, *args],
            cwd=self.tmp,
            capture_output=True,
            text=True,
        )

    def parse_r(self, stdout):
        """按考官现场取数的方式，从 stdout 里找 'r = <数值>' 这一行"""
        for line in stdout.splitlines():
            if line.strip().startswith("r ="):
                return float(line.split("=", 1)[1].strip())
        self.fail("stdout 里没有找到 'r = <数值>' 这一行，实际输出：\n%s" % stdout)

    def assertNoTraceback(self, proc):
        combined = proc.stdout + proc.stderr
        self.assertNotIn(
            "Traceback", combined, "不允许把栈回溯抛给用户，实际输出：\n%s" % combined
        )


class TestCliSuccess(CliTestCase):
    """正常路径：exit 0，且 stdout 必须有一行 r = <数值> 供提取比对"""

    def test_config_option(self):
        self.write_csv(GOOD_CSV)
        self.write_yaml("data.csv")
        proc = self.run_cli("--config", "config.yaml")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNoTraceback(proc)
        self.assertAlmostEqual(self.parse_r(proc.stdout), 1.0, places=6)

    def test_positional_argument(self):
        """题面：除 --config 外也支持位置参数更好"""
        self.write_csv(GOOD_CSV)
        self.write_yaml("data.csv")
        proc = self.run_cli("config.yaml")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertAlmostEqual(self.parse_r(proc.stdout), 1.0, places=6)

    def test_absolute_config_path(self):
        """--config 传绝对路径也要能跑通"""
        self.write_csv(GOOD_CSV)
        cfg = self.write_yaml("data.csv")  # write() 返回绝对路径
        proc = self.run_cli("--config", cfg)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertAlmostEqual(self.parse_r(proc.stdout), 1.0, places=6)


class TestCliAbnormalInput(CliTestCase):
    """
    题面：异常输入必须打印可读的错误信息、以非 0 退出码结束，且不得抛栈回溯。
    退出码按 README 里的分类表钉死——改动退出码表时这里要同步改。
    """

    def assertFails(self, proc, expect_code):
        self.assertEqual(
            proc.returncode,
            expect_code,
            "期望退出码 %s，实际 %s\n--- stdout ---\n%s\n--- stderr ---\n%s"
            % (expect_code, proc.returncode, proc.stdout, proc.stderr),
        )
        self.assertNoTraceback(proc)
        self.assertTrue(
            (proc.stdout + proc.stderr).strip(), "异常时必须给出可读的错误提示"
        )

    def test_config_file_not_found(self):
        self.assertFails(self.run_cli("--config", "nope.yaml"), 1)

    def test_config_path_is_a_directory(self):
        os.mkdir(os.path.join(self.tmp, "adir"))
        self.assertFails(self.run_cli("--config", "adir"), 1)

    def test_csv_file_not_found(self):
        self.write("nc.yaml", 'input_csv: "nope.csv"\ncolumns:\n  x: "sensor_a"\n  y: "sensor_b"\n')
        self.assertFails(self.run_cli("--config", "nc.yaml"), 1)

    def test_yaml_syntax_error(self):
        self.write("bad.yaml", 'input_csv: "data.csv"\ncolumns:\n  x: "a"\n   y: "b"\n')
        self.assertFails(self.run_cli("--config", "bad.yaml"), 6)

    def test_missing_columns_key(self):
        self.write_csv(GOOD_CSV)
        self.write("nc.yaml", 'input_csv: "data.csv"\n')
        self.assertFails(self.run_cli("--config", "nc.yaml"), 2)

    def test_column_name_mismatch(self):
        self.write_csv(GOOD_CSV)
        self.write_yaml("data.csv", col_y="not_a_column")
        self.assertFails(self.run_cli("--config", "config.yaml"), 2)

    def test_empty_csv(self):
        """只有表头、没有数据行"""
        self.write_csv("timestamp,sensor_a,sensor_b,note\n")
        self.write_yaml("data.csv")
        self.assertFails(self.run_cli("--config", "config.yaml"), 3)

    def test_non_numeric_cell(self):
        self.write_csv("timestamp,sensor_a,sensor_b,note\n1,abc,2,a\n")
        self.write_yaml("data.csv")
        self.assertFails(self.run_cli("--config", "config.yaml"), 3)

    def test_zero_variance(self):
        """某一列取值恒定 → 方差为 0 → 相关系数无定义"""
        self.write_csv("timestamp,sensor_a,sensor_b,note\n1,5,1,a\n2,5,2,b\n3,5,3,c\n")
        self.write_yaml("data.csv")
        self.assertFails(self.run_cli("--config", "config.yaml"), 3)

    def test_blank_yaml(self):
        """空 yaml：safe_load 返回 None，随后下标取值抛 TypeError"""
        self.write("blank.yaml", "")
        self.assertFails(self.run_cli("--config", "blank.yaml"), 4)

    def test_no_argument_at_all(self):
        """--config 和位置参数都不给：argparse 报错退出，且不得是栈回溯"""
        proc = self.run_cli()
        self.assertNotEqual(proc.returncode, 0)
        self.assertNoTraceback(proc)


if __name__ == "__main__":
    unittest.main()
