from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
import wandb
# import datasets
# from scipy.special import softmax


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train(out_dir,
          lm,
          trainset,
          valset,
          no_cuda,
          training_cfgs,
          project_name,
          run_name=None,
          save_model=True
          ) -> None:

    with wandb.init(project=project_name, name=run_name):
        if training_cfgs is None:
            # use wandb sweep config dict
            training_cfgs = wandb.config

        early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=training_cfgs['STOPPING_PATIENCE'])

        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            no_cuda=no_cuda,
            num_train_epochs=training_cfgs['MAX_EPOCHS'],
            per_device_train_batch_size=training_cfgs['TRAIN_BATCHSIZE'],
            per_device_eval_batch_size=training_cfgs['EVAL_BATCHSIZE'],
            gradient_accumulation_steps=training_cfgs['BATCH_UPDATE'],
            do_eval=True,
            evaluation_strategy=transformers.IntervalStrategy.EPOCH,
            learning_rate=training_cfgs['LR'],
            adam_epsilon=training_cfgs['ADAM_EPS'],
            weight_decay=training_cfgs['WEIGHT_DECAY'],
            save_total_limit=1,
            save_strategy=transformers.IntervalStrategy.EPOCH,
            load_best_model_at_end=True,
            metric_for_best_model=compute_metrics
        )

        trainer = transformers.Trainer(
            model=lm,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=valset,
            callbacks=[early_stopping]
        )

        trainer.train()

        if save_model:
            trainer.save_model()



# def main():
#
#     label_map = {0: "negative",
#                  1: "positive"}
#     d = {"text": ["I do not like this movie.", "I really like the movie."],
#          "labels": [0, 1]}
#     df_train = pd.DataFrame(d)
#
#     # tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
#     tokenizer = transformers.AutoTokenizer.from_pretrained('distilroberta-base')
#     lm = transformers.AutoModelForSequenceClassification.from_pretrained('distilroberta-base')
#
#     # create Hugging dataset
#     trainset = datasets.Dataset.from_pandas(df_train)
#     tokenized_train = trainset.map(lambda examples: tokenizer(examples["text"],
#                                                               padding="max_length",
#                                                               truncation=True), batched=True)
#
#     early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=1)
#     training_args = transformers.TrainingArguments(
#         output_dir="/home/diego/counterfactuals-generation/cata_sentiment/",
#         overwrite_output_dir=True,
#         no_cuda=False,
#         num_train_epochs=1,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=2,
#         do_eval=True,
#         evaluation_strategy=transformers.IntervalStrategy.EPOCH,
#         warmup_steps=1,
#         learning_rate=0.0001,
#         adam_epsilon=0.000011,
#         weight_decay=0.00001,
#         save_total_limit=1,
#         save_strategy=transformers.IntervalStrategy.EPOCH,
#         load_best_model_at_end=True,
#         metric_for_best_model='eval_loss'
#     )
#
#     trainer = transformers.Trainer(
#         model=lm,
#         args=training_args,
#         train_dataset=tokenized_train,
#         eval_dataset=tokenized_train,
#         callbacks=[early_stopping]
#     )
#
#     trainer.train()
#
#     lm = trainer.model
#
#     # precict
#     encoded_input = tokenizer("I like this.", return_tensors='pt')
#     output = lm(**encoded_input)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#
#     ranking = np.argsort(scores)
#     ranking = ranking[::-1]
#     for i in range(scores.shape[0]):
#         l = label_map[ranking[i]]
#         s = scores[ranking[i]]
#         print(f"{i+1}) {l} {np.round(float(s), 4)}")
#
#     # print(f"{label_map[scores[0]]}")


# if __name__ == "__main__":
#     main()
