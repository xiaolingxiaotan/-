# 参数说明

| 参数 | 含义 | 默认值 |
| ---- | ---- | ---- |
| --config | 任务配置 yaml 文件路径 | tasks_demo.yaml |
| --timeout | 全局总耗时上限 (浮点数，单位秒) | 读取 yaml 配置内的 timeout |
| --report | 输出报告 json 文件路径 | report.json |
| --seed | 随机数种子，固定种子可复现随机结果 | None（每次随机不同） |

# 日志输出说明
程序在控制台实时输出彩色日志，不同状态对应固定颜色：
- `SUCCESS`：绿色
- `FAILED`：红色
- `RETRY`：黄色
- `SKIPPED`：蓝色
- `TIMEOUT`：红色
其余普通文本为白色。

# 运行示例
## 示例一：使用原始提供的tasks_demo.yaml，超时时间按照配置文件中的15s，固定种子42
- 命令行输入：
```bash
python3 scheduler.py --config tasks_demo.yaml --seed 42
```
- 日志输出：
```
=== dependencies列表无重复 ===
=== 未检测到有向环 ===
2026-10-05 15:55:58,422 - INFO - 任务 Init 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:55:58,923 - INFO - 任务 Navigate 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:56:01,425 - INFO - 任务 DetectQR 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:56:02,927 - INFO - 任务 AvoidObstacle 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:56:04,929 - INFO - 任务 Grasp 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:56:05,930 - INFO - 任务 Place 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:56:06,931 - INFO - 任务 ReturnHome 执行成功，尝试次数：1，状态：SUCCESS
=== report导出成功 ===
```
- 生成的report:`/home/xiaolingxiaotan/智能车一轮考核/智能车一轮考核/M0/M0-4/report.json`
```json
{
  "timeout": false,
  "total_duration": 11.0,
  "tasks": [
    {
      "name": "Init",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 0.5,
      "started_at": 1791186958.4223673,
      "ended_at": 1791186958.4223695
    },
    {
      "name": "Navigate",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 2.5,
      "started_at": 1791186958.9230297,
      "ended_at": 1791186958.9230318
    },
    {
      "name": "DetectQR",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 1.5,
      "started_at": 1791186961.425692,
      "ended_at": 1791186961.4256945
    },
    {
      "name": "AvoidObstacle",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 2.0,
      "started_at": 1791186962.9273167,
      "ended_at": 1791186962.9273186
    },
    {
      "name": "Grasp",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 1.0,
      "started_at": 1791186964.9294546,
      "ended_at": 1791186964.9294574
    },
    {
      "name": "Place",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 1.0,
      "started_at": 1791186965.9306152,
      "ended_at": 1791186965.930618
    },
    {
      "name": "ReturnHome",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 2.5,
      "started_at": 1791186966.931796,
      "ended_at": 1791186966.9317987
    }
  ]
}
```

## 示例二，修改配置文件中AvoidObstacle任务的概率为0，超时时间按照配置文件中的15s，固定种子42
- 命令行输入：
```bash
python3 scheduler.py --config tasks_demo.yaml --seed 42
```
- 日志输出：
```
=== 未检测到有向环 ===
2026-10-05 15:57:12,371 - INFO - 任务 Init 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:57:12,872 - INFO - 任务 Navigate 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:57:15,374 - INFO - 任务 DetectQR 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:57:16,876 - INFO - 任务 AvoidObstacle 准备重试，当前尝试次数：1，状态：RETRY
2026-10-05 15:57:18,878 - INFO - 任务 AvoidObstacle 准备重试，当前尝试次数：2，状态：RETRY
2026-10-05 15:57:20,880 - INFO - 任务 AvoidObstacle 执行失败，尝试次数：3，状态：FAILED
2026-10-05 15:57:22,882 - INFO - 任务 Grasp 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:57:23,884 - INFO - 任务 Place 依赖不满足，任务跳过，状态：SKIPPED
2026-10-05 15:57:23,884 - INFO - 任务 ReturnHome 依赖不满足，任务跳过，状态：SKIPPED
=== report导出成功 ===
```
- 生成的report:`/home/xiaolingxiaotan/智能车一轮考核/智能车一轮考核/M0/M0-4/report.json`
```json
{
  "timeout": false,
  "total_duration": 11.5,
  "tasks": [
    {
      "name": "Init",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 0.5,
      "started_at": 1791187032.3712888,
      "ended_at": 1791187032.3712907
    },
    {
      "name": "Navigate",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 2.5,
      "started_at": 1791187032.8719163,
      "ended_at": 1791187032.871918
    },
    {
      "name": "DetectQR",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 1.5,
      "started_at": 1791187035.3745778,
      "ended_at": 1791187035.37458
    },
    {
      "name": "AvoidObstacle",
      "status": "FAILED",
      "attempts": 3,
      "duration": 2.0,
      "started_at": 1791187040.8805733,
      "ended_at": 0
    },
    {
      "name": "Grasp",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 1.0,
      "started_at": 1791187042.88275,
      "ended_at": 1791187042.882752
    },
    {
      "name": "Place",
      "status": "SKIPPED",
      "attempts": 0,
      "duration": null,
      "started_at": null,
      "ended_at": null
    },
    {
      "name": "ReturnHome",
      "status": "SKIPPED",
      "attempts": 0,
      "duration": null,
      "started_at": null,
      "ended_at": null
    }
  ]
}
```

## 示例三，使用原始提供的tasks_demo.json，超时时间按照配置文件中的15s，不固定种子
- 命令行输入：
```bash
python3 scheduler.py --config tasks_demo.json
```
- 日志输出：
```
=== dependencies列表无重复 ===
=== 未检测到有向环 ===
2026-10-05 15:58:30,145 - INFO - 任务 Init 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:58:30,646 - INFO - 任务 Navigate 准备重试，当前尝试次数：1，状态：RETRY
2026-10-05 15:58:33,148 - INFO - 任务 Navigate 准备重试，当前尝试次数：2，状态：RETRY
2026-10-05 15:58:35,651 - INFO - 任务 Navigate 执行成功，尝试次数：3，状态：SUCCESS
2026-10-05 15:58:38,154 - INFO - 任务 DetectQR 准备重试，当前尝试次数：1，状态：RETRY
2026-10-05 15:58:39,656 - INFO - 任务 DetectQR 执行成功，尝试次数：2，状态：SUCCESS
2026-10-05 15:58:41,157 - INFO - 任务 AvoidObstacle 准备重试，当前尝试次数：1，状态：RETRY
2026-10-05 15:58:43,159 - INFO - 任务 AvoidObstacle 执行成功，尝试次数：2，状态：SUCCESS
2026-10-05 15:58:45,161 - ERROR - 任务 Grasp 全局超时终止,状态：TIMEOUT
2026-10-05 15:58:45,162 - ERROR - 任务 Place 全局超时终止,状态：TIMEOUT
2026-10-05 15:58:45,162 - ERROR - 任务 ReturnHome 全局超时终止,状态：TIMEOUT
=== report导出成功 ===
```
- 生成的report:``
```json
{
  "timeout": true,
  "total_duration": 15,
  "tasks": [
    {
      "name": "Init",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 0.5,
      "started_at": 1791187110.1455276,
      "ended_at": 1791187110.1455286
    },
    {
      "name": "Navigate",
      "status": "SUCCESS",
      "attempts": 3,
      "duration": 2.5,
      "started_at": 1791187115.6514487,
      "ended_at": 1791187115.6514513
    },
    {
      "name": "DetectQR",
      "status": "SUCCESS",
      "attempts": 2,
      "duration": 1.5,
      "started_at": 1791187119.6558359,
      "ended_at": 1791187119.6558409
    },
    {
      "name": "AvoidObstacle",
      "status": "SUCCESS",
      "attempts": 2,
      "duration": 2.0,
      "started_at": 1791187123.159658,
      "ended_at": 1791187123.1596613
    },
    {
      "name": "Grasp",
      "status": "TIMEOUT",
      "attempts": 1,
      "duration": 1.0,
      "started_at": 1791187125.1617787,
      "ended_at": 1791187125.1617787
    },
    {
      "name": "Place",
      "status": "TIMEOUT",
      "attempts": 0,
      "duration": null,
      "started_at": null,
      "ended_at": null
    },
    {
      "name": "ReturnHome",
      "status": "TIMEOUT",
      "attempts": 0,
      "duration": null,
      "started_at": null,
      "ended_at": null
    }
  ]
}
```

## 示例四，使用原始提供的tasks_demo.yaml，超时时间为5s，模拟时间不够，固定种子42
- 命令行输入：
```bash
python3 scheduler.py --config tasks_demo.yaml --timeout 5 --seed 42
```
- 日志输出：
```
=== dependencies列表无重复 ===
=== 未检测到有向环 ===
2026-10-05 15:59:36,426 - INFO - 任务 Init 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:59:36,927 - INFO - 任务 Navigate 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:59:39,429 - INFO - 任务 DetectQR 执行成功，尝试次数：1，状态：SUCCESS
2026-10-05 15:59:40,931 - ERROR - 任务 AvoidObstacle 全局超时终止,状态：TIMEOUT
2026-10-05 15:59:41,432 - ERROR - 任务 Grasp 全局超时终止,状态：TIMEOUT
2026-10-05 15:59:41,432 - ERROR - 任务 Place 全局超时终止,状态：TIMEOUT
2026-10-05 15:59:41,432 - ERROR - 任务 ReturnHome 全局超时终止,状态：TIMEOUT
=== report导出成功 ===
```
- 生成的report:``
```json
{
  "timeout": true,
  "total_duration": 5.0,
  "tasks": [
    {
      "name": "Init",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 0.5,
      "started_at": 1791187176.4265623,
      "ended_at": 1791187176.4265673
    },
    {
      "name": "Navigate",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 2.5,
      "started_at": 1791187176.9271798,
      "ended_at": 1791187176.9271817
    },
    {
      "name": "DetectQR",
      "status": "SUCCESS",
      "attempts": 1,
      "duration": 1.5,
      "started_at": 1791187179.4298131,
      "ended_at": 1791187179.4298158
    },
    {
      "name": "AvoidObstacle",
      "status": "TIMEOUT",
      "attempts": 1,
      "duration": 2.0,
      "started_at": 1791187180.931436,
      "ended_at": 1791187180.931436
    },
    {
      "name": "Grasp",
      "status": "TIMEOUT",
      "attempts": 0,
      "duration": null,
      "started_at": null,
      "ended_at": null
    },
    {
      "name": "Place",
      "status": "TIMEOUT",
      "attempts": 0,
      "duration": null,
      "started_at": null,
      "ended_at": null
    },
    {
      "name": "ReturnHome",
      "status": "TIMEOUT",
      "attempts": 0,
      "duration": null,
      "started_at": null,
      "ended_at": null
    }
  ]
}
```

## 示例五，故意构造有向环（添加place->Grasp），使用原始提供的tasks_demo.yaml，超时时间按照配置文件中的15s，固定种子42
- 命令行输入：
```bash
python3 scheduler.py --config tasks_demo.yaml --seed 42
```
- 日志输出：
```
=== dependencies列表无重复 ===
捕获到有向环['Grasp', 'Place', 'Grasp']
```
- 生成的report:无

# AI 使用说明

**代码部分**：`scheduler.py` 的主体逻辑由本人独立编写，包括参数解析、配置加载、
Kahn 拓扑排序、失败重试与依赖状态传播、日志线程与 report 导出。

**AI 参与修复了以下四个 bug：**

### bug 1：异常输入不退出，且会打印栈回溯

- **现象**：除依赖成环外，其余异常输入（文件不存在、YAML/JSON 语法错误、
  空任务列表、`success_rate` 越界、依赖不存在的任务）程序都静默结束，
  退出码为 0；其中"依赖了不存在的任务"与"配置缺少 `timeout`"两处
  还会直接抛出 `KeyError` 栈回溯。
- **原因**：`main` 的 `except` 分支里只有 `logging.error(...)` 而没有 `sys.exit`，
  异常被接住后函数正常返回，进程退出码仍然是 0；
  兜底分支写了 `exc_info=True`，把栈回溯一并打印了出来。
- **修复**：约定统一退出码（1 文件 / 2 有向环 / 3 语法错误 / 4 配置内容非法）；
  每个 `except` 分支补上非 0 退出；去掉 `exc_info=True`；
  新增 `config_check` 在入口统一拦截非法配置。

### bug 2：有向环检测是死代码，环永远检测不到

- **现象**：故意构造 `Place -> Grasp` 成环后运行，程序既不报错也不退出，
  直接当成正常配置往下走。
- **原因**：`DFS` 里"已访问则剪枝"的判断（`if u in visited: return`）排在了
  "回到当前递归路径即成环"的判断（`if u in in_stack: raise`）之前，
  而 `visited.add(u)` 在节点入栈时就已执行，于是 `in_stack` 恒为 `visited`
  的子集——成环节点会先命中剪枝返回，成环分支永远不可达。
- **修复**：把成环判断调到剪枝判断之前，并删掉没有用到的 `cycle` 参数；
  环路径由 `DFS` 抛出的 `ValueError` 携带，`kahn` 捕获后打印成
  `A -> B -> A` 的可读形式并以退出码 2 结束。

### bug 3：超时控制不严——撞线的任务会睡过头，失败重试还不计入耗时

- **现象**：`--timeout 5` 时，report 里的 `total_duration` 会出现 6.5；
  把某个任务的 `success_rate` 设为 0 让它重试 3 次，实际墙钟耗时明显超过
  配置的超时上限却没有被拦住。
- **原因**：两处。一是 `sleep(duration)` 睡的是任务的完整预期耗时，没有按
  "距离超时线还剩多少"截断，所以越过超时线的那一次会睡满整个 `duration`；
  二是 `total_duration` 只在任务**成功**的分支里累加，失败重试的那几次
  `sleep` 完全没进预算，剩余额度越算越大，超时也就触发不了。
- **修复**：改成预算式钳制——每次尝试前先算剩余额度
  `remaining = TIME_OUT - total_duration["time"]`，实际只睡
  `min(duration, remaining)`；每次 `sleep` 之后无条件累加并钳制在
  `TIME_OUT` 以内；一旦发现睡不满计划时长（说明撞线了），
  立即判 `TIMEOUT` 且不再重试。

### bug 4：双线程协调时日志事件被吞

- **现象**：主线程与日志线程之间用共享 list 传递任务状态事件，
  出现事件丢失 / 竞态，部分任务状态没有被打印出来。
- **修复**：改用 `deque` 作为事件队列，配合 `threading.Condition` 的
  `wait` / `notify_all` 做同步——主线程 `append`、日志线程 `popleft`，
  生产与消费解耦，不再互相覆盖。

以上四处修复的方案由 AI 给出，本人逐行理解并验证了修改后的实际行为：
六个异常输入逐个复跑确认退出码与错误提示，构造有向环确认能报错退出，
并用正常配置、`--timeout` 中断、`success_rate` 置 0 三条路径做了回归。