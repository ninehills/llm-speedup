# Speculative decoding

```bash
cd SpecForge/
# 不安装依赖，SpecForge 依赖很乱
pip install -v . --no-deps
# 准备英文数据
python scripts/prepare_data.py --dataset ultrachat
# python scripts/prepare_data.py --dataset sharegpt

# 准备中文数据
python -c "import datasets,uuid;\
  ds=datasets.load_dataset('Congliu/Chinese-DeepSeek-R1-Distill-data-110k', split='train')\
  .filter(lambda example: example['score'] >= 10)\
  .map(lambda example: {'id': str(uuid.uuid4()), 'conversations': [{'role': 'user', 'content': example['input']}, {'role': 'assistant', 'content': example['content']}]})\
  .select_columns(['id', 'conversations']);\
  ds.to_json('cache/dataset/deepseek-distill-data.jsonl', orient='records', lines=True, force_ascii=False)"

# 各选1w条
shuf cache/dataset/deepseek-distill-data.jsonl | head -n 10000 > cache/dataset/train.jsonl
shuf cache/dataset/ultrachat_train.jsonl | head -n 10000 >> cache/dataset/train.jsonl

# 训练
export TORCHINDUCTOR_CACHE_DIR=cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=1

# 可以自己根据母模型的配置文件，修改 draft-model-config，让除了 layers 等参数保持一致 
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_online.py \
    --target-model-path ../Qwen3-4B-Instruct-2507/ \
    --draft-model-config ./configs/qwen3-4b-eagle3.json \
    --train-data-path cache/dataset/train.jsonl \
    --output-dir outputs/Qwen3-4B-Instruct-2507-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir ./cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --ttt-length 7
```

