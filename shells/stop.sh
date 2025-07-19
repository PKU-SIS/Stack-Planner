#!/bin/bash

cd ..
# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sp

# 查找并终止服务器进程
echo "正在停止服务器..."
SERVER_PID=$(ps -ef | grep "python server.py" | grep -v grep | awk '{print $2}')

if [ -z "$SERVER_PID" ]; then
  echo "没有找到运行中的服务器进程"
else
  echo "终止进程 PID: $SERVER_PID"
  kill -9 $SERVER_PID
  echo "服务器已停止"
fi
