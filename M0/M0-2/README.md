# 如何运行
- 1，准备环境：安装 python3、pyyaml、numpy
- 2，准备好配置文件yaml
- 3，支持 --config访问，请在终端中执行python3 /home/xiaolingxiaotan/智能车一轮考核/智能车一轮考核/M0/M0-2/legacy_corr.py --config config.yaml
- 4，支持位置参数访问，请在终端中执行python3 /home/xiaolingxiaotan/智能车一轮考核/智能车一轮考核/M0/M0-2/legacy_corr.py config.yaml

# 输入文件格式要求
- **1，yaml内必须包含：**
input_csv: "xxx.csv"       # csv文件路径
columns:
  x: "x_data"              # x列在csv中的列名
  y: "y_data"              # y列在csv中的列名
- **2，csv中必须包含：**
1. 第一行为表头，包含yaml指定的x、y两列名称
2. 后续每行是数值，数据可以转为浮点数
3. 至少一行有效数据；不能只有表头无数据
4. 文件编码为 UTF-8

# 输出结果的含义
- 1，n表示列的长度，即有效样本的个数
- 2，mean_x 与 mean_y表示x，y两列的数据的平均值
- 3，r表示相关系数，表示两列数据的相关性，取值范围为[-1.1]，绝对值越接近1表示两列数据的相关性越强
- 4，numpy验证信息：对比手写计算结果与numpy库计算结果，误差小于1e-6判定验证成功
- 5，把两列打印出散点图便于看趋势和相关性
- 6，可能出现的异常现象说明，其中不同的异常类型退出码不同，具体退出码如下表：

| 退出码 | 类别 | 说明 |
|:---:|:---:|---|
| 0 | 代码无异常 | 程序正常执行完成 |
| 1 | IO / 文件相关 | 文件不存在、权限不足、路径为目录、文件解码失败等操作系统IO类错误 |
| 2 | 字典、索引、取值类 | KeyError、IndexError，字典键缺失、列表下标越界、列名不匹配 |
| 3 | 数值、计算类 | ZeroDivisionError、ValueError，除零、方差为0、数值非法、空数据集 |
| 4 | 类型相关 | TypeError，参数类型不匹配，无法执行运算 |
| 5 | 导入、模块相关 | ModuleNotFoundError，依赖库未安装（如numpy） |
| 6 | 语法、运行基础类 | SyntaxError、IndentationError，代码语法错误（程序启动前触发） |
| 7 | 用户中断 | KeyboardInterrupt，用户按下 `Ctrl+C` 终止程序 |
| 8 | 第三方库自定义异常 | yaml.YAMLError等第三方库抛出的专属异常 |

# AI 使用说明

**主程序部分**：`legacy_corr.py` 的全部代码由本人独立编写，未使用 AI 生成。
yaml 配置解析、csv 读取、求和 / 均值 / 相关系数计算、异常分类与退出码设计、
numpy 交叉验证、散点图绘制，均为本人逐行手写；开发过程中遇到的报错与数值异常
也由本人自行定位和解决。

**测试部分**：`tests/` 目录下的单元测试（`tests/test_corr.py`、`tests/test_cli.py`）
由 AI 生成初稿，本人逐行读懂并验证后才纳入交付。

**验证方式**：AI 给出测试后，本人做了两步交叉验证，确认这些用例真的在测东西，
而不是跑了个空：

1. **全绿**：`python3 -m unittest discover -s tests -v`，28 个用例全部通过。
2. **会红**——故意把 `legacy_corr.py` 的关键行改坏，确认对应用例真的会失败：
   - 把 `r_calc` 里的 `dx = dx + a * a` 改成 `dx = dx + a * b`：
     28 个用例中 12 个失败（9 个断言失败 + 3 个错误）；
   - 把 `r_calc` 的 `return r` 改成 `return abs(r)`：
     `test_perfect_negative_correlation` 等 2 个用例失败。

   只有能变红的测试才说明它在真的断言；改坏后依然全绿的用例等于没测。

**测试设计上本人确认过的两个判断**（AI 初稿即按此写，本人复核认可）：

- **正确性用例的期望值不手算**，改用 `numpy.corrcoef` 现算作为独立参照；另加
  "对调 x/y 不变、平移与正比例缩放不变"的性质用例——公式里错一个符号，
  性质用例往往比单个数值更容易抓到。
- **异常用例一律用 `subprocess` 跑整个 CLI**，不在测试进程内直接调函数：
  退出码分支是靠 `sys.exit` 结束的，`SystemExit` 在测试进程内一触发会把整个
  测试跑带停；而且"stdout 必须含 `r = <数值>`"、"不得出现 `Traceback`"
  这两条题面要求，只有在真实进程的 stdout / stderr 上才验得了。

