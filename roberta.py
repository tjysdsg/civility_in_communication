# https://huggingface.co/docs/transformers/tasks/sequence_classification
from datasets import Dataset
from data import load_train_df, load_dev_df, load_test_df, df2dataset, load_demographic_dev_df
import evaluate
from transformers import AutoTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np
from sklearn.metrics import classification_report


# from ekphrasis.classes.preprocessor import TextPreProcessor
# from ekphrasis.classes.tokenizer import SocialTokenizer
# from ekphrasis.dicts.emoticons import emoticons


# https://github.com/cbaziotis/ekphrasis
# text_processor = TextPreProcessor(
#     # terms that will be normalized
#     normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
#     # terms that will be annotated
#     annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
#     fix_html=True,  # fix HTML tokens
#     # corpus from which the word statistics are going to be used for word segmentation
#     segmenter="twitter",
#     # corpus from which the word statistics are going to be used for spell correction
#     corrector="twitter",
#     unpack_hashtags=True,  # perform word segmentation on hashtags
#     unpack_contractions=True,  # Unpack contractions (can't -> can not)
#     spell_correct_elong=False,  # spell correction for elongated words
#     # select a tokenizer. You can use SocialTokenizer, or pass your own
#     # the tokenizer, should take as input a string and return a list of tokens
#     tokenizer=SocialTokenizer(lowercase=True).tokenize,
#     # list of dictionaries, for replacing tokens extracted from the text,
#     # with other expressions. You can pass more than one dictionaries.
#     dicts=[emoticons]
# )


# def preprocess_social_text(text):
#     res = []
#     for t in text:
#         res.append(" ".join(text_processor.pre_process_doc(t)))
#     return res


def tokenize_dataset(data: Dataset, tokenizer):
    # return data.map(lambda x: tokenizer(preprocess_social_text(x["text"]), truncation=True), batched=True)
    return data.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)


def main():
    train_set = df2dataset(load_train_df())
    dev_set = df2dataset(load_dev_df())
    demo_set = df2dataset(load_demographic_dev_df())

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
    ).to('cuda')

    if args.train:
        train_set = tokenize_dataset(train_set, tokenizer)
    else:
        train_set = None
    dev_set = tokenize_dataset(dev_set, tokenizer)
    demo_set = tokenize_dataset(demo_set, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="exp",
        learning_rate=1e-5,
        per_device_train_batch_size=28,
        per_device_eval_batch_size=28,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to='none',
        fp16=True,
        metric_for_best_model='eval_accuracy',
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
    # {'eval_loss': 0.45169317722320557, 'eval_accuracy': 0.7930513595166163, 'epoch': 1.0}

    y_pred = trainer.predict(dev_set).predictions.argmax(axis=-1)

    y_dev = dev_set['label']
    print(f'\nDEV SET:\n{classification_report(y_dev, y_pred)}')

    y_demo_pred = trainer.predict(demo_set).predictions.argmax(axis=-1)

    fp = y_demo_pred.sum()
    tn = len(y_demo_pred) - fp
    print(f'FPR: {fp / (fp + tn)}')
    """
    DEV SET:
                  precision    recall  f1-score   support

               0       0.83      0.87      0.85       884
               1       0.71      0.65      0.68       440

        accuracy                           0.80      1324
       macro avg       0.77      0.76      0.76      1324
    weighted avg       0.79      0.80      0.79      1324

    FPR: 0.1474763406940063
    """


if __name__ == '__main__':
    main()
