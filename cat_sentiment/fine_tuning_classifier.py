from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
import wandb
import yaml
import os
import argparse
import datetime
import datasets
from sentiment_task import utils


def clean_dataset(df, name):
    # remove the null values (if any)
    n_nan = df['text'].isna().sum()
    print(f"# of nan values removed in {name}:{n_nan}")
    df.dropna(inplace=True)
    return df


def prepare_training(df_train, df_val, tokenizer, batch_tokens) -> (datasets.Dataset, datasets.Dataset):
    trainset = datasets.Dataset.from_pandas(df_train)
    valset = datasets.Dataset.from_pandas(df_val)

    tokenized_train = trainset.map(lambda examples: tokenizer(examples["text"],
                                                              padding="max_length",
                                                              truncation=True), batched=batch_tokens)
    tokenized_val = valset.map(lambda examples: tokenizer(examples["text"],
                                                          padding="max_length",
                                                          truncation=True), batched=batch_tokens)

    return tokenized_train, tokenized_val


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

    # with wandb.init(project=project_name, name=run_name):
    wandb.init(project=project_name, name=run_name)

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
        metric_for_best_model='accuracy'
    )

    trainer = transformers.Trainer(
        model=lm,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    trainer.train()

    if save_model:
        trainer.save_model(out_dir)

    print(trainer.evaluate())


def main():
    # read params from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting_path",
        default=None,
        type=str,
        required=True,
        help="The absolute path of the file settings."
    )

    parser.add_argument(
        "--setting_name",
        default=None,
        type=str,
        required=True,
        help="The name of yaml file where to load the setting from."
    )

    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        required=True,
        help="The path of the dataset to use."
    )

    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        help="The name of the dataset to use."
    )

    parser.add_argument(
        "--wandb_key",
        default=None,
        type=str,
        required=True,
        help="The API key of wandb used to login."
    )

    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        required=True,
        help="The project path in wandb."
    )

    parser.add_argument(
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )

    args = parser.parse_args()

    # read params from yaml file
    setting_yaml_file = open(f"{args.setting_path}{args.setting_name}")
    parsed_yaml_file = yaml.load(setting_yaml_file, Loader=yaml.FullLoader)

    print(f"{datetime.datetime.now()}: Begin tuning for dataset:{args.dataset_name}")

    val_prop = parsed_yaml_file['VAL_PROP']
    out_dir = parsed_yaml_file['OUT_DIR']

    lm_name = parsed_yaml_file['LM_NAME']
    tokenize_in_batch = parsed_yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = parsed_yaml_file['NO_CUDA']

    random_seed = parsed_yaml_file['RANDOM_SEED']
    print("Tuning params read from yaml file")

    # load the dataset
    df_trainset, df_valset = utils.load_dataset_with_val(random_seed,
                                                         val_prop,
                                                         f"{args.dataset_path}/{args.dataset_name}.csv"
                                                         )

    # remove the null values (if any)
    df_trainset = clean_dataset(df_trainset, "trainset")
    df_valset = clean_dataset(df_valset, "valset")

    if args.debug_mode:
        df_trainset = df_trainset[:10]
        df_valset = df_valset[:10]
    print(f"# of samples for training:{len(df_trainset)}")
    print(f"# of samples for validation:{len(df_valset)}")

    # roberta-base\distilroberta-base have a model_max_length of 512
    tokenizer = transformers.AutoTokenizer.from_pretrained(lm_name)
    lm = transformers.AutoModelForSequenceClassification.from_pretrained(lm_name,
                                                                         num_labels=2
                                                                         )

    print("Downloaded tokenizer, model and cfg!")

    tokenized_train, tokenized_val = prepare_training(df_trainset,
                                                      df_valset,
                                                      tokenizer,
                                                      tokenize_in_batch)
    print("Datasets have been tokenized successfully!")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    training_cfgs = parsed_yaml_file['TRAINING_CFGS']
    dataset_name = parsed_yaml_file['DATASET_NAME']
    out_label = parsed_yaml_file['OUT_LABEL']

    out_name = f"{lm_name}@{dataset_name}@{out_label}"
    out_path = f"{out_dir}/{out_name}"

    train(out_path,
          lm,
          tokenized_train,
          tokenized_val,
          no_cuda,
          training_cfgs,  # training cfgs
          args.wandb_project,
          out_name,  # run_name
          True  # save_model
          )

    wandb.finish()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

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
