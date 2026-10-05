import argparse
import os
import yaml
import json
import sys
import collections
import random
import threading
import time
import logging
from collections import deque

# 定义日志颜色类
class ColoredFormatter(logging.Formatter):        #logging.Formatter是日志格式化器的基类
    reset = "\033[0m"    #重置颜色
    color_map = {
        "SUCCESS": "\033[32m",           #绿色
        "FAILED": "\033[31m",            #红色
        "RETRY": "\033[33m",             #黄色
        "SKIPPED": "\033[34m",           #蓝色
        "TIMEOUT": "\033[31m"            #红色
    }
    # 重写格式函数
    def format(self, record):  
        msg_full = super().format(record)
        for word, color_code in self.color_map.items():
            msg_full = msg_full.replace(word, f"{color_code}{word}{self.reset}")
        return msg_full

# 解析命令行并得到调度器的初始配置
def scheduler_init():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="任务调度器")
    parser.add_argument("--config",default="tasks_demo.yaml",help="任务配置文件地址")
    parser.add_argument("--timeout",default= None,type=float,help="全局总耗时上限")
    parser.add_argument("--report",default="report.json",help="输出报告路径")
    parser.add_argument("--seed",type=int,default= None,help="随机数种子值")
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
        elif choice == ".json":
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
    task_deque = collections.deque()
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
    # 拓扑排序没能排出全部任务，说明剩下的部分里一定有环，交给DFS找出来
    visited = set(task_list)    # 已处理完的节点，可直接剪枝
    in_stack = set()            # 当前递归路径上的节点
    path = []
    for u in all_names:
        if u not in visited:
            DFS(visited,in_stack,path,u,out_edges)
            
# 辅助函数，递归DFS搜索
def DFS(visited,in_stack,path,u,out_edges):
    if u in in_stack:
        idx = path.index(u)
        raise ValueError(path[idx:] + [u])
    if u in visited:
        return
    in_stack.add(u)
    path.append(u)
    visited.add(u)
    try:
        for dp in out_edges[u]:
            DFS(visited,in_stack,path,dp,out_edges)
     # 回溯，便于递归下一个路径回路
    finally:
        in_stack.remove(u)
        path.pop()

# 任务调度器        
def task_goingon(TIME_OUT,tasks,task_list,SEED,conlock,task_status,log_queue):
    if SEED is None:
        rad = random.Random()
    else:
        rad = random.Random(SEED)
    ntk = name_to_task(tasks)
    total_duration = {"time":0}
    default = {"name":None,"status":None,"attempts":0,"duration":0.0,"started_at": 0,"ended_at": 0}
    global_timeout = False
    total_status = {}
    
    for t in task_list:
        deps = ntk[t]["dependencies"]
        all_dep_ok = True
        for dep in deps:
            if dep not in total_status or total_status[dep]["status"] != "SUCCESS":
                all_dep_ok = False
                break
        # 依赖任务成功执行且前一个任务未超时
        if all_dep_ok and global_timeout == False:
            with conlock:
                task_status.update(default)
            timeout=last_success(rad,conlock,task_status,default,ntk,t,TIME_OUT,total_duration,log_queue)
            if timeout:
                global_timeout = True
            total_status[t] = task_status.copy()

        # 依赖任务执行失败且前一个任务未超时
        elif not all_dep_ok and global_timeout == False:
            with conlock:
                task_status.update(default)
                task_status["name"] = t
                task_status["attempts"] = 0
                task_status["duration"] = None
                task_status["started_at"] = None
                task_status["ended_at"] = None
                total_duration["time"] += 0
                task_status["status"] = "SKIPPED"
                timeout = False
                log_queue.append(task_status.copy())
                conlock.notify_all()
            total_status[t] = task_status.copy()  

        # 前一个任务已超时
        elif global_timeout == True:
            with conlock:
                task_status.update(default)
                task_status["name"] = t
                task_status["attempts"] = 0
                task_status["duration"] = None
                task_status["started_at"] = None
                task_status["ended_at"] =None
                total_duration["time"] += 0
                task_status["status"] = "TIMEOUT"
                timeout = True
                log_queue.append(task_status.copy())
                conlock.notify_all()
            total_status[t] = task_status.copy() 

    return global_timeout,total_status,total_duration


# 辅助函数：创建名字查询任务字典，避免每次遍历
def name_to_task(tasks):
    ntk = {t["name"]: t for t in tasks}
    return ntk

# 辅助函数：依赖函数正常执行时操作
def last_success(rad,conlock,task_status,default,ntk,t,TIME_OUT,total_duration,log_queue):
    max_retry = 3
    for attempt in range(1, max_retry+1):
        task_finished = False
        # 剩余额度：本任务最多只睡到这条线为止
        remaining = TIME_OUT - total_duration["time"]
        sleep_for = min(ntk[t]["duration"], remaining)
        if sleep_for < 0:
            sleep_for = 0
        # 睡不满计划时长，说明这一次尝试撞上了全局上限
        time_is_up = sleep_for < ntk[t]["duration"]
        with conlock:
            task_status["name"] = t
            task_status["attempts"] = attempt
            task_status["duration"] = ntk[t]["duration"]
            task_status["started_at"] = time.time()
            if time_is_up:
                # 撞线：本任务到此为止，直接判超时，不再重试
                task_status["ended_at"] = time.time()
                task_status["status"] = "TIMEOUT"
                log_queue.append(task_status.copy())
                conlock.notify_all()
            else:
                r = rad.random()
                if r <= ntk[t]["success_rate"]:
                    # 任务成功
                    task_status["ended_at"] = time.time()
                    task_status["status"] = "SUCCESS"
                    log_queue.append(task_status.copy())
                    conlock.notify_all()
                    task_finished = True
                else:
                    # 失败，判断是否还能重试
                    if attempt < max_retry:
                        task_status["status"] = "RETRY"
                        log_queue.append(task_status.copy())
                        conlock.notify_all()
                    else:
                        # 3次全部失败
                        task_status["status"] = "FAILED"
                        log_queue.append(task_status.copy())
                        conlock.notify_all()
        # 只睡到上限：撞线时刚好睡到 timeout 为止
        time.sleep(sleep_for)
        # 每次实际 sleep 都累加耗时，并把总值钳制在上限内
        total_duration["time"] = min(TIME_OUT, total_duration["time"] + sleep_for)
        if time_is_up:
            return True
        if task_finished:
            return False
    return False


            
# 打印日志函数
def logging_print(task_status,conlock,stop_flag,log_queue):
    while True:
        snap = None
        with conlock:
            # 等待：要么停止，要么有新事件
            while (not stop_flag[0]) and (len(log_queue) == 0):
                conlock.wait(timeout=0.1)
            # 退出条件
            if stop_flag[0] and len(log_queue) == 0:
                break
            snap = log_queue.popleft() 
            conlock.notify_all()
        if snap is not None:
            name = snap["name"]
            status = snap["status"]
            attempts = snap["attempts"]
            if status == "SUCCESS":
                logging.info(f"任务 {name} 执行成功，尝试次数：{attempts}，状态：SUCCESS")
            elif status == "FAILED":
                logging.info(f"任务 {name} 执行失败，尝试次数：{attempts}，状态：FAILED")
            elif status == "RETRY":
                logging.info(f"任务 {name} 准备重试，当前尝试次数：{attempts}，状态：RETRY")
            elif status == "SKIPPED":
                logging.info(f"任务 {name} 依赖不满足，任务跳过，状态：SKIPPED")
            elif status == "TIMEOUT":
                logging.error(f"任务 {name} 全局超时终止,状态：TIMEOUT")

def report_out(REPORT_PATH,total_status,total_duration,global_timeout):
    report = {"timeout":global_timeout,"total_duration":total_duration["time"],"tasks":list(total_status.values())}
    with open(REPORT_PATH,"w",encoding="utf-8") as f:
        choice=os.path.splitext(REPORT_PATH)[1].lower() # 将路径的后缀提取出来并转为小写
        if choice in(".yaml",".yml"):
            yaml.safe_dump(report,f,encoding="utf-8")
        elif choice == ".json":
            json.dump(report,f,indent=2,ensure_ascii=False)
        else:
            print("配置文件格式不对")
            sys.exit(1)
    print("=== report导出成功 ===")

def main():
    # 初始化logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 新增定义，防止finally检测不到变量报错
    conlock = None
    log = None
    try:
        CONFIG_PATH,global_timeout,REPORT_PATH,SEED=scheduler_init()
        TIME_OUT,tasks=file_getting(CONFIG_PATH,global_timeout)
        task_list=kahn(tasks)
        task_status = {"name":None,"status":None,"attempts":0,"duration":0.0,"started_at": 0,"ended_at": 0}
        conlock = threading.Condition(lock= None)
        stop_flag = [False]
        log_queue = deque()
        log=threading.Thread(target=logging_print,name=None,args=(task_status,conlock,stop_flag,log_queue),daemon=False)
        log.start()
        global_timeout,total_status,total_duration = task_goingon(TIME_OUT,tasks,task_list,SEED,conlock,task_status,log_queue)
        with conlock:
            while log_queue:
                conlock.wait()
        with conlock:
            stop_flag[0] = True
            conlock.notify_all()
        log.join() 
        report_out(REPORT_PATH,total_status,total_duration,global_timeout)

    except FileNotFoundError:
        logging.error("异常：找不到配置文件，请检查--config路径")
    except yaml.YAMLError:
        logging.error("异常：yaml配置文件解析失败，格式错误")
    except json.JSONDecodeError:
        logging.error("异常：json配置文件解析失败，格式错误")
    except ValueError as e:
        logging.error(f"值异常：{e}")
    except KeyboardInterrupt:
        logging.warning("收到Ctrl+C中断，准备退出")
    except Exception as e:
        logging.error(f"未知运行异常: {type(e).__name__}: {e}", exc_info=True)
    finally:
        if conlock is not None and log is not None and log.is_alive():
            with conlock:
                stop_flag[0] = True
                conlock.notify_all()
            log.join()
    
if __name__ == "__main__":
    main()
    