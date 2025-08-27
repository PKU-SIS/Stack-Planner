cd ..
# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sp

nohup python server.py --reload --host 0.0.0.0 --port 8082 > logs/server.log 2>&1 &