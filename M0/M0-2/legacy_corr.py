# 导入库
import yaml
import csv
import math
import argparse

#设置路径
def PATH_Setting():
    parser = argparse.ArgumentParser(description="分析csv文件并算出相关系数")
    parser.add_argument("-c", "--config", help="yaml配置文件路径")
    parser.add_argument("config_path",nargs="?",help="配置文件位置参数")
    args = parser.parse_args()
    CONFIG_PATH = args.config if args.config is not None else args.config_path
    if CONFIG_PATH is None:
        parser.error("yaml文件路径输入错误")
    return CONFIG_PATH
                   
#解析数据函数
def Data_analysis(xs,ys,CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)       #解析基础类型：字典、列表、数字、字符串、bool、null
    csv_path = cfg["input_csv"]
    col_x = cfg["columns"]["x"]
    col_y = cfg["columns"]["y"]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row[col_x]))
            ys.append(float(row[col_y]))
        n=len(xs)
    return n

# 求和x与y
def sum(xs,ys,n):
    sum_x = 0.0
    sum_y = 0.0
    for i in range(n):
        sum_x = sum_x + xs[i]
        sum_y = sum_y + ys[i]
    return sum_x,sum_y

# 计算x与y的均值
def mean(xs,ys,n):
    sum_x,sum_y=sum(xs,ys,n)
    mean_x = sum_x / n
    mean_y = sum_y / n
    return mean_x,mean_y

# 计算分母项与分子项 ，加上开根号
def r_calc(xs,ys,n):
    dx = 0.0
    dy = 0.0
    prod = 0.0
    mean_x,mean_y= mean(xs,ys,n)
    for i in range(n):
        a = xs[i] - mean_x
        b = ys[i] - mean_y
        dx = dx + a * a
        dy = dy + b * b
        prod = prod + a * b
    denom = math.sqrt(dx) *math.sqrt(dy) 
    r = prod / denom
    return r

# 打印出数据
def data_print(xs,ys,n):
    print("n =", n)
    print("mean_x =", mean(xs,ys,n)[0])
    print("mean_y =", mean(xs,ys,n)[1])
    print("r =",r_calc(xs,ys,n) )

def main():
    CONFIG_PATH=PATH_Setting()
    xs = []
    ys = []
    Data_analysis(xs,ys,CONFIG_PATH)
    n = len(xs)
    r_calc(xs,ys,n)
    data_print(xs,ys,n)

if __name__ == "__main__":
    main()
