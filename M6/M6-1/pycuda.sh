#!/usr/bin/env bash
# 给 onnxruntime 找 CUDA 运行库：torch 自带的 nvidia/*/lib 不在默认搜索路径里
VENV="/home/xiaolingxiaotan/智能车一轮考核/智能车一轮考核/.venv"
SP="$VENV/lib/python3.10/site-packages"
EXTRA=""
for d in "$SP"/nvidia/*/lib; do
    [ -d "$d" ] && EXTRA="$EXTRA:$d"
done
export LD_LIBRARY_PATH="$EXTRA${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$VENV/bin/python" "$@"
