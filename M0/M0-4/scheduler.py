import argparse
import os
import yaml
import json
import sys

# 解析命令行并得到调度器的初始配置
def scheduler_init():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="任务调度器")
    parser.add_argument("--config",default="tasks_demo.yaml",help="任务配置文件地址")
    parser.add_argument("--timeout",default= 15,help="全局总耗时上限")
    parser.add_argument("--report",default="report.json",help="输出报告路径")
    parser.add_argument("--seed",default= None,help="随机数种子值")
    args = parser.parse_args()
    CONFIG_PATH=os.path.join(script_dir,args.config)
    TIME_OUT=args.timeout
    REPORT_PATH=os.path.join(script_dir,args.report)
    SEED=args.seed
    return CONFIG_PATH,TIME_OUT,REPORT_PATH,SEED

# 根据路径读取解析文件
def file_getting(CONFIG_PATH):
    with open(CONFIG_PATH,"r",encoding="utf-8") as f:
        choice=os.path.splitext(CONFIG_PATH)[1].lower() # 将路径的后缀提取出来并转为小写
        if choice in(".yaml",".yml"):
            data=yaml.safe_load(f)
        elif choice == "json":
            data = json.load(f)
        else:
            print("配置文件格式不对")
            sys.exit(1)
    return data

def main():
    CONFIG_PATH,TIME_OUT,REPORT_PATH,SEED=scheduler_init()
    data=file_getting(CONFIG_PATH)
    print(data)

if __name__ == "__main__":
    main()