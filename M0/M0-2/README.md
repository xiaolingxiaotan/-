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
- 5，可能出现的异常现象说明，其中不同的异常类型退出码不同，具体退出码如下表：

|退出码	|类别	|说明
|0	|代码无异常	|程序正常执行完成
|1	|IO / 文件相关	|文件不存在、权限不足、路径为目录、文件解码失败等操作系统 IO 类错误
|2	|字典、索引、取值类	|KeyError、IndexError，字典键缺失、列表下标越界、列名不匹配
|3	|数值、计算类	|ZeroDivisionError、ValueError，除零、方差为 0、数值非法、空数据集
|4	|类型相关	|TypeError，参数类型不匹配，无法执行运算
|5	|导入、模块相关	|ModuleNotFoundError，依赖库未安装（如 numpy）
|6	|语法、运行基础类	|SyntaxError、IndentationError，代码语法错误（程序启动前触发）
|7	|用户中断	|KeyboardInterrupt，用户按下 Ctrl+C 终止程序
|8	|第三方库自定义异常	|yaml.YAMLError 等第三方库抛出的专属异常
