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