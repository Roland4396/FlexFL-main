#!/bin/bash
# FlexFL 批量训练脚本 - 让GPU不要空闲
# 训练3个模型 x 2种数据分布 x 3种Non-IID程度 = 18个实验

# 基础配置
GPU=0
DATASET="cifar10"
NUM_CHANNELS=3
NUM_CLASSES=10
CLIENT_RATIO="1:1:1:1"
PRETRAIN=200
GAMMA=10
ONLY=1

# 输出目录
OUTPUT_BASE="outputs"
mkdir -p $OUTPUT_BASE

# 记录开始时间
echo "=========================================="
echo "批量训练开始: $(date)"
echo "=========================================="

# 模型列表
MODELS=("vgg" "mobilenet" "resnet")

# 数据分布: IID + 3种Non-IID程度
declare -A DISTRIBUTIONS=(
    ["iid"]="--iid 1"
    ["noniid_high"]="--iid 0 --data_beta 10"
    ["noniid_medium"]="--iid 0 --data_beta 50"
    ["noniid_low"]="--iid 0 --data_beta 100"
)

# 实验计数
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0

# 计算总实验数
for model in "${MODELS[@]}"; do
    for dist_name in "${!DISTRIBUTIONS[@]}"; do
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    done
done

echo "总共需要运行 $TOTAL_EXPERIMENTS 个实验"
echo "=========================================="

# 开始批量训练
for model in "${MODELS[@]}"; do
    for dist_name in "${!DISTRIBUTIONS[@]}"; do
        COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))

        # 构建输出路径
        OUTPUT_PATH="${OUTPUT_BASE}/${model}_${dist_name}"

        echo ""
        echo "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] 开始训练:"
        echo "  模型: $model"
        echo "  数据分布: $dist_name"
        echo "  输出路径: $OUTPUT_PATH"
        echo "  开始时间: $(date)"
        echo "------------------------------------------"

        # 获取分布参数
        DIST_PARAMS=${DISTRIBUTIONS[$dist_name]}

        # 构建完整命令
        CMD="python main_fed.py \
            --gpu $GPU \
            --algorithm FlexFL \
            --model $model \
            --dataset $DATASET \
            --num_channels $NUM_CHANNELS \
            --num_classes $NUM_CLASSES \
            $DIST_PARAMS \
            --client_hetero_ration $CLIENT_RATIO \
            --client_chosen_mode available \
            --pretrain $PRETRAIN \
            --gamma $GAMMA \
            --only $ONLY \
            2>&1 | tee ${OUTPUT_PATH}.log"

        # 显示命令
        echo "执行命令:"
        echo "$CMD"
        echo ""

        # 执行训练
        eval $CMD

        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "✓ 训练完成: $model - $dist_name"
        else
            echo "✗ 训练失败: $model - $dist_name"
            echo "  错误信息请查看: ${OUTPUT_PATH}.log"
        fi

        echo "  结束时间: $(date)"
        echo "=========================================="

        # 短暂休息（可选，避免过热）
        sleep 5
    done
done

# 记录结束时间
echo ""
echo "=========================================="
echo "批量训练完成: $(date)"
echo "完成 $COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS 个实验"
echo "所有日志保存在: $OUTPUT_BASE/"
echo "=========================================="

# 生成训练报告
echo ""
echo "生成训练报告..."
echo "实验结果汇总:" > ${OUTPUT_BASE}/summary.txt
echo "生成时间: $(date)" >> ${OUTPUT_BASE}/summary.txt
echo "========================================" >> ${OUTPUT_BASE}/summary.txt

for model in "${MODELS[@]}"; do
    echo "" >> ${OUTPUT_BASE}/summary.txt
    echo "模型: $model" >> ${OUTPUT_BASE}/summary.txt
    echo "----------------------------------------" >> ${OUTPUT_BASE}/summary.txt
    for dist_name in "${!DISTRIBUTIONS[@]}"; do
        LOG_FILE="${OUTPUT_BASE}/${model}_${dist_name}.log"
        if [ -f "$LOG_FILE" ]; then
            # 提取最后的准确率（如果有）
            LAST_ACC=$(grep -oP "test_acc.*?[\d\.]+%" "$LOG_FILE" | tail -1 || echo "未找到")
            echo "  $dist_name: $LAST_ACC" >> ${OUTPUT_BASE}/summary.txt
        else
            echo "  $dist_name: 日志文件不存在" >> ${OUTPUT_BASE}/summary.txt
        fi
    done
done

echo "训练报告已保存: ${OUTPUT_BASE}/summary.txt"
echo "所有任务完成！"
