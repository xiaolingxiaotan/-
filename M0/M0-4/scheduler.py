import argparse
import os
import yaml
import json
import sys
import collections

# 解析命令行并得到调度器的初始配置
def scheduler_init():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="任务调度器")
    parser.add_argument("--config",default="tasks_demo.yaml",help="任务配置文件地址")
    parser.add_argument("--timeout",default= None,help="全局总耗时上限")
    parser.add_argument("--report",default="report.json",help="输出报告路径")
    parser.add_argument("--seed",default= None,help="随机数种子值")
    args = parser.parse_args()
    CONFIG_PATH=os.path.join(script_dir,args.config)
    timeout=args.timeout
    REPORT_PATH=os.path.join(script_dir,args.report)
    SEED=args.seed
    return CONFIG_PATH,timeout,REPORT_PATH,SEED

# 根据路径读取解析文件
def file_getting(CONFIG_PATH,timeout):
    with open(CONFIG_PATH,"r",encoding="utf-8") as f:
        choice=os.path.splitext(CONFIG_PATH)[1].lower() # 将路径的后缀提取出来并转为小写
        if choice in(".yaml",".yml"):
            data=yaml.safe_load(f)
        elif choice == "json":
            data = json.load(f)
        else:
            print("配置文件格式不对")
            sys.exit(1)
    if timeout == None:
        TIME_OUT = data["timeout"]
    else: 
        TIME_OUT = timeout
    tasks = data["tasks"]
    return TIME_OUT,tasks

# Kahn算法拓扑排序
def kahn(tasks):
    # 初始化
    all_names = [t["name"] for t in tasks]
    in_degree = {name:0 for name in all_names}
    out_edges = {name:[] for name in all_names}
    task_deque = collections.deque(iterable=None,maxlen=None)
    task_list = []

    # 查看每个任务的dependencies列表，明确列表无重复（后续遍历操作不会出错）
    kahn_duplicate_check(tasks)

    # 统计入度出边
    for task in tasks:
        in_degree[task["name"]] = len(task["dependencies"])
        child_task = task["name"]
        for parent_task in task["dependencies"]:
            out_edges[parent_task].append(child_task)

    # 按排序加入队列
    for task in tasks:
        if in_degree[task["name"]] == 0:
            task_deque.append(task["name"])
    while len(task_deque) != 0:
        t = task_deque.popleft()
        task_list.append(t)
        for i in out_edges[t]:
            in_degree[i] -= 1
            if in_degree[i] == 0:
                task_deque.append(i)

    # 有向环检测
    try:
        kahn_cycle_detection(task_list,all_names,out_edges)
    except ValueError as e:
        print(f"捕获到有向环{e}")
        sys.exit(2)

    return task_list

# 辅助函数，检测 dependencies列表无重复              
def kahn_duplicate_check(tasks):
    error_duplicate = []
    for task in tasks:
        if len(set(task["dependencies"])) != len(task["dependencies"]):
            seen = set()
            for dep in task["dependencies"]:
                if dep in seen:
                    error_duplicate.append(dep)
                seen.add(dep)
            print("在{}中的依赖任务列表中发现重复任务{}".format(task["name"],error_duplicate))
    print("=== dependencies列表无重复 ===")
    return error_duplicate

# 辅助函数，检测有向环
def kahn_cycle_detection(task_list,all_names,out_edges):
    if len(task_list) == len(all_names):
        print("=== 未检测到有向环 ===")
        return
    else:    #进行DFS找环
        visited = set(task_list)
        in_stack = set()
        path = []
        cycle = None
        for u in all_names:
            if u not in visited:
                DFS(visited,in_stack,path,cycle,u,out_edges)
            
# 辅助函数，递归DFS搜索
def DFS(visited,in_stack,path,cycle,u,out_edges):
    if u in visited:
        return
    if u in in_stack:
        idx = path.index(u)
        cycle = path[idx:]
        cycle.append(u)
        raise ValueError(cycle)
    in_stack.add(u)
    path.append(u)
    visited.add(u)
    try:
        for dp in out_edges[u]:
            DFS(visited,in_stack,path,cycle,dp,out_edges)
     # 回溯，便于递归下一个路径回路
    finally:
        in_stack.remove(u)
        path.pop()
        


def main():
    CONFIG_PATH,timeout,REPORT_PATH,SEED=scheduler_init()
    TIME_OUT,tasks=file_getting(CONFIG_PATH,timeout)


if __name__ == "__main__":
    main()