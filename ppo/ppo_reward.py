from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from torch.optim import RMSprop
import torch
dataset_name = 'imdb'
model_name = "gpt2-large"

reward_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# inputs = tokenizer(batch_input_texts, return_tensors="pt", padding=True, truncation=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 生成提示文本并生成标签
def tokenize_prompts(reviews):
    inputs = tokenizer(reviews, return_tensors="pt", padding=True, truncation=True, max_length=512)
    labels = inputs.input_ids.detach().clone()  # 将 input_ids 复制为 labels
    inputs['labels'] = labels
    return inputs

train_data = load_dataset(dataset_name, split="train")
train_prompts = [" ".join(review.split()[:4]) for review in train_data["text"]]
train_inputs = tokenize_prompts(train_prompts)

test_data = load_dataset(dataset_name, split="test")
test_prompts = [" ".join(review.split()[:4]) for review in test_data["text"]]
test_inputs = tokenize_prompts(test_prompts)

# 转换为Dataset对象
train_dataset = Dataset.from_dict({
    "input_ids": train_inputs["input_ids"],
    "attention_mask": train_inputs["attention_mask"],
    "labels": train_inputs["labels"]
})
test_dataset = Dataset.from_dict({
    "input_ids": test_inputs["input_ids"],
    "attention_mask": test_inputs["attention_mask"],
    "labels": test_inputs["labels"]
})
# breakpoint()
# 自定义优化器
def custom_optimizer(model):
    return RMSprop(model.parameters(), lr=1e-6)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=150,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=1e-6,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=(custom_optimizer(reward_model), None),  # 自定义优化器
)

trainer.train()
# 保存最佳模型
trainer.save_model("./results/best_model")  # 训练结束后保存最佳模型