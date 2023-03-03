import pandas as pd
from datasets import Dataset
from data import load_train_df, load_dev_df, load_test_df, df2dataset
import evaluate
from transformers import AutoTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np


def tokenize_dataset(data: Dataset, tokenizer):
    return data.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)


def preprocess(df: pd.DataFrame):
    df['label'] = (df['label'] == 'OFF').astype(int)
    return df


def train():
    train_set = df2dataset(preprocess(load_train_df()))
    dev_set = df2dataset(preprocess(load_dev_df()))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
    ).to('cuda')

    train_set = tokenize_dataset(train_set, tokenizer)
    dev_set = tokenize_dataset(dev_set, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="exp",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def main():
    train()


if __name__ == '__main__':
    main()
