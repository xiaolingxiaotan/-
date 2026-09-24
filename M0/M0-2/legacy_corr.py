# 导入库
import yaml
import csv
import math

#设置路径
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)       #解析基础类型：字典、列表、数字、字符串、bool、null

csv_path = cfg["input_csv"]
col_x = cfg["columns"]["x"]
col_y = cfg["columns"]["y"]

# 将数据存入列表
xs = []
ys = []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        xs.append(float(row[col_x]))
        ys.append(float(row[col_y]))

n = len(xs)

# 求和x与y
sum_x = 0.0
sum_y = 0.0
for i in range(n):
    sum_x = sum_x + xs[i]
    sum_y = sum_y + ys[i]

# 计算x与y的均值
mean_x = sum_x / n
mean_y = sum_y / n

# 计算分母项与分子项 ：分母项缺少开根号，逻辑出错
dx = 0.0
dy = 0.0
prod = 0.0
for i in range(n):
    a = xs[i] - mean_x
    b = ys[i] - mean_y
    dx = dx + a * a
    dy = dy + b * b
    prod = prod + a * b

# 计算r
denom = dx * dy
r = prod / denom

# 打印出数据
print("n =", n)
print("mean_x =", mean_x)
print("mean_y =", mean_y)
print("r =", r)
