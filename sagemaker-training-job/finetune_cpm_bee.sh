#! /bin/bash

export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True 
pip install -r requirements.txt

if [ $? -eq 1 ]; then
    echo "pip install error, please check CloudWatch logs"
    exit 1
fi

chmod +x ./s5cmd
./s5cmd sync s3://$MODEL_S3_BUCKET/llm/models/cpm-bee/10B/* /tmp/cpm_pretrain/
./s5cmd sync s3://$MODEL_S3_BUCKET/llm/datasets/cpm-bee/bin_data/* /tmp/bin_data/

#单卡微调
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12346

OPTS=""
OPTS+=" --use-delta"  # 使用增量微调（delta-tuning）
OPTS+=" --model-config CPM-Bee/src/config/cpm-bee-10b.json"  # 模型配置文件
OPTS+=" --dataset /tmp/bin_data/train"  # 训练集路径
OPTS+=" --eval_dataset /tmp/bin_data/eval"  # 验证集路径
OPTS+=" --epoch 5"  # 训练epoch数
OPTS+=" --batch-size 8"    # 数据批次大小
OPTS+=" --train-iters 100"  # 用于lr_schedular
OPTS+=" --save-name cpm_bee_finetune"  # 保存名称
OPTS+=" --max-length 1024" # 最大长度
OPTS+=" --save /tmp/results/"  # 保存路径
OPTS+=" --lr 0.0001"    # 学习率
OPTS+=" --inspect-iters 100"  # 每100个step进行一次检查(bmtrain inspect)
OPTS+=" --warmup-iters 1" # 预热学习率的步数为1
OPTS+=" --eval-interval 50"  # 每50步验证一次
OPTS+=" --early-stop-patience 5"  # 如果验证集loss连续5次不降，停止微调
OPTS+=" --lr-decay-style noam"  # 选择noam方式调度学习率
OPTS+=" --weight-decay 0.01"  # 优化器权重衰减率为0.01
OPTS+=" --clip-grad 1.0"  # 半精度训练的grad clip
OPTS+=" --loss-scale 32768"  # 半精度训练的loss scale
OPTS+=" --start-step 0"  # 用于加载lr_schedular的中间状态
OPTS+=" --load /tmp/cpm_pretrain/pytorch_model.bin"  # 模型参数文件


CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} CPM-Bee/src/finetune_cpm_bee.py ${OPTS}"

echo ${CMD}
$CMD

if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi

./s5cmd sync /tmp/results s3://$MODEL_S3_BUCKET/cpm-bee/output/$(date +%Y-%m-%d-%H-%M-%S)/
