

#!/bin/bash

# 定义工作目录
WORK_DIR="/fs-computility/llmit_d/shared/liumengjie/LLaMA-Factory"

# 定义节点列表（根据你的实际节点名称/IP地址进行修改）
NODES=("node1" "node2" "node3" "node4")

# 循环遍历节点并执行命令
for i in {0..3}; do
  NODE=${NODES[$i]}
  echo "在节点 $NODE 上启动训练 (NODE_RANK=$i)..."
  
  if [ "$i" -eq 0 ]; then
    # 在本地执行主节点命令
    cd $WORK_DIR && bash $WORK_DIR/cmds/multi_node/train_bash.sh html 42 0 &
  else
    # 在远程节点上执行命令
    ssh $NODE "cd $WORK_DIR && bash $WORK_DIR/cmds/multi_node/train_bash.sh html 42 $i" &
  fi
done

echo "所有节点的训练任务已启动!"
wait
echo "所有训练任务已完成!"

