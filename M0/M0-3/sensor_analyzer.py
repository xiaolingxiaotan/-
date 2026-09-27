#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensor_analyzer.py  —— 上一届学长留下的"能用"的脚本

注释（学长原话）：
    "处理一下传感器数据就能用"

原本意图：
    1. 读取 sensor_data.csv（列：time, value）
    2. 计算 value 的平均值、标准差
    3. 剔除离群值（|value - mean| > 2 * std）
    4. 把清洗后的数据保存为 cleaned_data.csv
    5. 打印一份统计摘要

现状：跑不通 / 跑出来数不对。就交给你了。
"""

"""
退出码约定：(沿用M0-2)
0 ———— 代码无异常
1 ———— IO / 文件相关
2 ———— 字典、索引、取值类
3 ———— 数值、计算类
4 ———— 类型相关
5 ———— 导入、模块相关
6 ———— 语法、运行基础类
7 ———— 用户中断
8 ———— 第三方库自定义异常
"""

import csv
import os
import math
import argparse
import sys

# --- 路径读取 ---
def path_setting():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="传感器数据清洗")
    parser.add_argument("--input", default="sensor_data.csv", help="输入csv路径，默认当前目录sensor_data.csv")
    parser.add_argument("--output", default="cleaned_data.csv", help="输出csv路径，默认当前目录cleaned_data.csv")
    args = parser.parse_args()
    INPUT_FILE = os.path.join(script_dir, args.input)
    OUTPUT_DIR = os.path.join(script_dir, "out")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, args.output)
    return INPUT_FILE,OUTPUT_DIR, OUTPUT_FILE 

# --- 读取数据 ---
def reader(times,data,INPUT_FILE):
    print("=== 传感器数据分析 ===")
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(open(INPUT_FILE, "r"))

        for row in reader:
            t = float(row["time"])
            v = float(row["value"])
            times.append(t)
            data.append(v)

    print("共读取 %d 条数据" % len(data))

# --- 计算平均值 ---
def mean_calc(data):
    total = 0
    for v in data:
        total += v
    mean = total / len(data)
    return mean

# --- 计算标准差 ---
def std_calc(mean,data):
    acc = 0
    for v in data:
        acc += (v - mean)**2
    std = math.sqrt(acc / len(data))
    return std

# --- 剔除离群值 ---
def remove_outliers(mean,std,times,data,cleaned):
    for i in range(len(data)):
        v = data[i]
        t = times[i]
        if abs(v - mean) <= std*2:
            cleaned.append( [t, v] )

# --- 输出清洗后的数据 ---
def result_output(cleaned,OUTPUT_FILE):
    out_path = OUTPUT_FILE
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "value"])
        writer.writerows(cleaned)
    return out_path

def result_print(mean,std,cleaned,out_path):
    print("均值 mean = %.4f" % mean)
    print("标准差 std = %.4f" % std)
    print("清洗后剩余 %d 条" % len(cleaned))
    print("已保存到 %s" % out_path)

def main():
    INPUT_FILE,OUTPUT_DIR,OUTPUT_FILE=path_setting()
    data = []
    times = []
    cleaned = []
    reader(times,data,INPUT_FILE)
    if len(data) == 0:
        print("错误：读取到0条有效数据，无法计算")
        sys.exit(3)
    mean=mean_calc(data)
    std=std_calc(mean,data)
    remove_outliers(mean,std,times,data,cleaned)
    out_path = result_output(cleaned,OUTPUT_FILE)
    result_print(mean,std,cleaned,out_path)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except FileNotFoundError:
        print(f"错误：输入文件不存在")
        sys.exit(1)
    except PermissionError:
        print("错误：文件权限不足，无法读写")
        sys.exit(1)
    except KeyError as e:
        print(f"错误：CSV文件缺少列：{e}，需要time、value列")
        sys.exit(2)
    except ZeroDivisionError:
        print("错误：数据为空，除法计算失败")
        sys.exit(3)
    except ValueError:
        print("错误：CSV内存在无法转为数字的内容")
        sys.exit(4)
    except KeyboardInterrupt:
        print("\n程序被用户手动中断")
        sys.exit(7)
    except Exception as e:
        print(f"未知错误：{e}")
        sys.exit(8)

