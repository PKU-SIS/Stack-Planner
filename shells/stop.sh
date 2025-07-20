#!/bin/bash

cd ..
# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sp

# 查找并终止服务器进程
echo "正在停止服务器..."

# 方法1：通过端口查找进程
echo "查找监听8082端口的进程..."
PORT_PID=$(lsof -ti:8082)

if [ -n "$PORT_PID" ]; then
  echo "找到监听8082端口的进程 PID: $PORT_PID"
  echo "终止进程..."
  kill -9 $PORT_PID
  echo "已终止监听8082端口的进程"
else
  echo "没有找到监听8082端口的进程"
fi

# 方法2：通过进程名查找（备用方法）
echo "查找server.py相关进程..."
SERVER_PIDS=$(ps -ef | grep "[p]ython.*server.py" | awk '{print $2}')

if [ -n "$SERVER_PIDS" ]; then
  echo "找到以下server.py进程:"
  for PID in $SERVER_PIDS; do
    echo "终止进程 PID: $PID"
    kill -9 $PID
  done
  echo "所有server.py进程已停止"
else
  echo "没有找到运行中的server.py进程"
fi

# 验证是否还有残留进程
REMAINING=$(lsof -ti:8082)
if [ -n "$REMAINING" ]; then
  echo "警告：仍有进程监听8082端口 (PID: $REMAINING)"
else
  echo "验证完成：8082端口已释放"
fi
