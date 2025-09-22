from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

# Select model and load it.
model_id = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select number of samples. 1024 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 8192

# 使用zh和en的chat 数据集作为calibration dataset

eng_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")\
    .shuffle(seed=42)\
    .select(range(10000))\
    .map(lambda examples: {"text": tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
        )}, batched=True)\
    .filter(lambda example: len(example["text"]) <= MAX_SEQUENCE_LENGTH)\
    .select_columns(["text"])\
    .select(range(int(NUM_CALIBRATION_SAMPLES/2)))
print(eng_ds)

zh_ds = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k", split="train")\
    .shuffle(seed=42)\
    .filter(lambda example: example["score"] >= 10)\
    .select(range(10000))\
    .map(lambda example: {"messages": [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["content"]}
    ]})\
    .map(lambda examples: {"text": tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
        )}, batched=True)\
    .filter(lambda example: len(example["text"]) <= MAX_SEQUENCE_LENGTH)\
    .select_columns(["text"])\
    .select(range(int(NUM_CALIBRATION_SAMPLES/2)))
print(zh_ds)

ds = concatenate_datasets([eng_ds, zh_ds]).shuffle(seed=42)
print(f"Total calibration samples: {len(ds)}")
print(ds)

print(f"Calibration dataset saved to calibration.jsonl and calibration.txt")
ds.to_json("calibration.jsonl", orient="records", lines=True, force_ascii=False)

with open("calibration.txt", "w") as f:
    for example in ds:
        f.write(example["text"] + "\n")