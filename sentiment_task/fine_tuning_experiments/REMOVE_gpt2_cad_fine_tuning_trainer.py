import pandas as pd
import transformers
import datasets
import itertools
import argparse
import datetime
import yaml
# import pytorch_lightning
# from datasets import metric


def prova():
    # datasets.load_metric("perplexity")
    transformers.EarlyStoppingCallback


def load_dataset(loading_path):
    train = pd.read_csv(loading_path + "training_set", sep='\t')
    val = pd.read_csv(loading_path + "val_set", sep='\t')
    return train, val


def load_language_model_objects(model_name, spec_tokens):
    # Load language model objects
    tok = transformers.GPT2Tokenizer.from_pretrained(model_name)
    print("Downloaded tokenizer!")
    if spec_tokens is not None:
        print(f"Len of tokenizer before adding tokens:{len(tok)}")
        tok.add_special_tokens(spec_tokens) # add special tokens
        print("Added special tokens to tokenizer!")
        print(f"Len of tokenizer after adding tokens:{len(tok)}")

    model_config_class = transformers.GPT2Config.from_pretrained(model_name)
    model = transformers.GPT2LMHeadModel.from_pretrained(model_name, config=model_config_class)
    print("Downloaded model and cfg!")
    if spec_tokens is not None:
        # special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tok))

    return tok, model, model_config_class


def wrap_with_prompt(df_row, template, mapping_labels, spec_tokens):
    final_text = template.replace("<label_ex>", mapping_labels[df_row["label_ex"]])
    final_text = final_text.replace("<example_text>", df_row["example"])
    final_text = final_text.replace("<label_counter>", mapping_labels[df_row["label_counter"]])
    final_text = final_text.replace("<counter_text>", df_row["counterfactual"])
    final_text = final_text.replace("<sep>", spec_tokens["sep_token"])
    final_text = final_text.replace("<bos_token>", spec_tokens["bos_token"])
    final_text = final_text.replace("<eos_token>", spec_tokens["eos_token"])
    return final_text


def freeze_layers_lm(to_freeze, n_to_unfreeze, model):
    if to_freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for i, m in enumerate(model.transformer.h):
            # Only un-freeze the last n transformer blocks
            # dklas
            if i+1 > len(model.transformer.h) - n_to_unfreeze:
                for parameter in m.parameters():
                    parameter.requires_grad = True

        for parameter in model.transformer.ln_f.parameters():
            parameter.requires_grad = True

        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
        print(f"Freezed the first {len(model.transformer.h)-n_to_unfreeze} model's layers")
        print(f"Only the last {n_to_unfreeze} model's layers will be trained!")
    else:
        print("All the model's layers will be trained!")

    return model


def print_cfg(str_cfg):
    return ' '.join(str(el) for el in str_cfg)


def main():

    # read params from command line
    parser = argparse.ArgumentParser()
    # SETTINGS_PATH = "/home/diego/counterfactuals-generation/sentiment_task/fine_tuning_experiments/settings/"
    parser.add_argument(
        "--setting_path",
        default=None,
        type=str,
        required=True,
        help="The absolute path of the file settings."
    )

    # e.g. SETTING_NAME = "tuning_cad_prompt_1.yaml"
    parser.add_argument(
        "--setting_name",
        default=None,
        type=str,
        required=True,
        help="The name of yaml file where to load the setting from."
    )

    parser.add_argument(
        "--wandb_key",
        default=None,
        type=str,
        required=True,
        help="The API key of wandb."
    )

    args = parser.parse_args()

    # read params from yaml file
    setting_yaml_file = open(f"{args.setting_path}{args.setting_name}")
    parsed_yaml_file = yaml.load(setting_yaml_file, Loader=yaml.FullLoader)

    validation_path = parsed_yaml_file['VALIDATION_PATH']
    folds = parsed_yaml_file['FOLDS']
    dataset_path = parsed_yaml_file['DATASET_PATH']

    lm_name = parsed_yaml_file['LM_NAME']
    special_tokens = parsed_yaml_file['SPECIAL_TOKENS']
    tokenize_in_batch = parsed_yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = parsed_yaml_file['NO_CUDA']

    template_prompt = parsed_yaml_file['TEMPLATE_PROMPT']
    map_labels = parsed_yaml_file['MAP_LABELS']

    to_freeze_layers = parsed_yaml_file['TO_FREEZE_LAYERS']
    unfreeze_last_n = parsed_yaml_file['UNFREEZE_LAST_N']
    training_grid = parsed_yaml_file['TRAINING_GRID']

    # GEN_ARGS = parsed_yaml_file['GEN_ARGS']
    print("Training's params read from yaml file")

    all_pars = sorted(training_grid)
    train_grid = list(itertools.product(*(training_grid[par] for par in all_pars)))

    print(f"{datetime.datetime.now()}: BEGIN OF THE EXPERIMENTS")
    for fold in folds:
        print(f"{datetime.datetime.now()}: Begin of the experiments for fold:{fold}")
        for training_cfg in train_grid:
            print(f"{datetime.datetime.now()}: Training cfg: {training_cfg}")

            # load the dataset
            df_trainset, df_valset = load_dataset(f"{dataset_path}/fold_{fold}/")
            print(f"# of samples for training:{len(df_trainset)}")
            print(f"# of samples for validation:{len(df_valset)}")

            tokenizer, lm, lm_config_class = load_language_model_objects(lm_name, special_tokens)
            print("Downloaded tokenizer, model and cfg!")

            # wrap the datasets with the prompt template
            df_trainset["wrapped_input"] = df_trainset.apply(lambda row: wrap_with_prompt(row,
                                                                                          template_prompt,
                                                                                          map_labels,
                                                                                          special_tokens), axis=1)
            print("Training set wrapped!")
            df_valset["wrapped_input"] = df_valset.apply(lambda row: wrap_with_prompt(row,
                                                                                      template_prompt,
                                                                                      map_labels,
                                                                                      special_tokens), axis=1)
            print("Validation set wrapped!")

            # convert dataset from pandas to Dataset
            training_set = datasets.Dataset.from_pandas(df_trainset)
            val_set = datasets.Dataset.from_pandas(df_valset)

            # TOKENIZE datasets
            tokenized_train = training_set.map(lambda examples: tokenizer(examples["wrapped_input"],
                                                                          padding="max_length",
                                                                          truncation=True), batched=tokenize_in_batch)
            # tokenized_train.features['labels'] = tokenized_train.features['input_ids']
            tokenized_train = tokenized_train.add_column("labels", tokenized_train['input_ids'])
            tokenized_val = val_set.map(lambda examples: tokenizer(examples["wrapped_input"],
                                                                   padding="max_length",
                                                                   truncation=True), batched=tokenize_in_batch)
            # tokenized_val.features['labels'] = tokenized_val.features['input_ids']
            tokenized_val = tokenized_val.add_column("labels", tokenized_val['input_ids'])
            print("Datasets have been tokenized successfully!")

            lm = freeze_layers_lm(to_freeze_layers, unfreeze_last_n, lm)

            model_dir = f"{validation_path}fold_{fold}/gpt2-{print_cfg(training_cfg)}/"
            # training_cfg[0] = epochs
            # training_cfg[1] = train_batchsize
            # training_cfg[2] = eval_batchsize
            # training_cfg[3] = batch_update
            # training_cfg[4] = warmup_steps
            # training_cfg[5] = lr
            # training_cfg[6] = adam_epsilon
            # training_cfg[7] = weight_decay
            training_args = transformers.TrainingArguments(
                        output_dir=model_dir,
                        no_cuda=no_cuda,
                        num_train_epochs=training_cfg[0],
                        per_device_train_batch_size=training_cfg[1],
                        per_device_eval_batch_size=training_cfg[2],
                        gradient_accumulation_steps=training_cfg[3],
                        do_eval=True,
                        evaluation_strategy=transformers.IntervalStrategy.EPOCH,
                        warmup_steps=training_cfg[4],
                        learning_rate=training_cfg[5],
                        adam_epsilon=training_cfg[6],
                        weight_decay=training_cfg[7],
                        save_total_limit=1,
                        save_strategy=transformers.IntervalStrategy.EPOCH,
                        load_best_model_at_end=False
                    )

            trainer = transformers.Trainer(
                model=lm,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val
            )

            trainer.train()
            trainer.save_model()

            print(f"{datetime.datetime.now()}: End of experiments for cfg: {training_cfg}")

        print(f"{datetime.datetime.now()}: End of experiments for fold:{fold}")


if __name__ == "__main__":
    main()
