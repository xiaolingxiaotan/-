'''
异常退出码说明：
0 ———— 代码无异常
1 ———— IO / 文件相关
2 ———— 字典、索引、取值类
3 ———— 数值、计算类
4 ———— 类型相关
5 ———— 导入、模块相关
6 ———— 语法、运行基础类
7 ———— 用户中断
8 ———— 第三方库自定义异常
'''

# 导入库
import yaml
import csv
import math
import argparse
import numpy as np
import sys

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
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)       #解析基础类型：字典、列表、数字、字符串、bool、null
    except FileNotFoundError:
        print(f"错误：配置文件 {CONFIG_PATH} 不存在")
        sys.exit(1)
    except PermissionError:
        print(f"错误：无权限读取文件 {CONFIG_PATH}")
        sys.exit(1)
    except IsADirectoryError:
        print(f"错误：{CONFIG_PATH} 是文件夹，不是文件")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"错误：文件 {CONFIG_PATH} 编码不是UTF-8")
        sys.exit(3)
    except yaml.YAMLError:
        print(f"错误：yaml文件 {CONFIG_PATH} 语法错误，解析失败")
        sys.exit(6)
    try:
        csv_path = cfg["input_csv"]
        col_x = cfg["columns"]["x"]
        col_y = cfg["columns"]["y"]
    except KeyError as e:
        print(f"错误：列名不匹配{e}")
        sys.exit(2)
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    xs.append(float(row[col_x]))
                    ys.append(float(row[col_y]))
                except KeyError:
                    print(f"错误：列名不匹配")
                    sys.exit(2)
                except ValueError:
                    print(f"错误：数据无法转换为浮点数")
                    sys.exit(3)
            n=len(xs)
    except FileNotFoundError:
        print(f"错误：csv文件 {csv_path} 不存在")
        sys.exit(1)
    except PermissionError:
        print(f"错误：无权限读取csv文件 {csv_path}")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"错误：csv文件 {csv_path} 编码不是UTF-8")
        sys.exit(3)
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
def data_print(xs,ys,n,r):
    print("n =", n)
    print("mean_x =", mean(xs,ys,n)[0])
    print("mean_y =", mean(xs,ys,n)[1])
    print("r =",r )

# numpy的验证函数
def numpy_verity(xs,ys,r):
    arr_x = np.array(xs)
    arr_y = np.array(ys)
    corr_matrix = np.corrcoef(arr_x, arr_y)
    r_np = corr_matrix[0, 1]
    diff = abs(r - r_np)
    if diff < 1e-6:
        print("numpy验证成功 两者误差不超过1e-6")
    else:
        print("numpy验证失败 r=",r,"r_np=",r_np)

def main():
    try:
        CONFIG_PATH=PATH_Setting()
        xs = []
        ys = []
        Data_analysis(xs,ys,CONFIG_PATH)
        n = len(xs)
        if n <= 0:
            print("错误：CSV文件为空，没有有效数据")
            sys.exit(3)
        r = r_calc(xs,ys,n)
        numpy_verity(xs,ys,r)
        data_print(xs,ys,n,r)
        
    except KeyError as e:
        print(f"错误：列不存在：{e}")
        sys.exit(2)
    except IndexError:
        print("错误：数组下标越界")
        sys.exit(2)
    except ZeroDivisionError:
        print("错误：分母为0，无法计算相关系数（某一列全部数值相同）")
        sys.exit(3)
    except ValueError as e:
        print(f"数值错误：{e}")
        sys.exit(3)
    except TypeError:
        print("错误：类型不匹配，无法进行数值计算")
        sys.exit(4)
    except ModuleNotFoundError as e:
        print(f"错误：模块未安装：{e}")
        sys.exit(5)
    except Exception as e:
        print(f"未知异常：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
