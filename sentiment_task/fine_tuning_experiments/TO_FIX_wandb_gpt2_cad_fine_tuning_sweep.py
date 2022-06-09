import pandas as pd
import transformers
import datasets
import argparse
import datetime
import yaml
import wandb
import sys


def load_dataset(loading_path):
    trainset = pd.read_csv(loading_path + "training_set", sep='\t')
    val = pd.read_csv(loading_path + "val_set", sep='\t')
    return trainset, val


def load_language_model_objects(model_name, spec_tokens):
    # Load language model objects
    tok = transformers.GPT2Tokenizer.from_pretrained(model_name)
    print("Downloaded tokenizer!")
    if spec_tokens is not None:
        print(f"Len of tokenizer before adding tokens:{len(tok)}")
        tok.add_special_tokens(spec_tokens)  # add special tokens
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


def train(out_dir, model, trainset, valset, no_cuda):

    with wandb.init():
        config_dict = wandb.config
        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            no_cuda=no_cuda,
            num_train_epochs=config_dict["EPOCHS"],
            per_device_train_batch_size=config_dict["TRAIN_BATCHSIZE"],
            per_device_eval_batch_size=config_dict["EVAL_BATCHSIZE"],
            gradient_accumulation_steps=config_dict["BATCH_UPDATE"],
            do_eval=True,
            evaluation_strategy=transformers.IntervalStrategy.EPOCH,
            warmup_steps=config_dict["WARMUP_STEPS"],
            learning_rate=config_dict["LR"],
            adam_epsilon=config_dict["EPS"],
            weight_decay=config_dict["WEIGHT_DECAY"],
            save_total_limit=0,
            save_strategy='no',
            # save_strategy=transformers.IntervalStrategy.EPOCH,
            load_best_model_at_end=False
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=valset
        )

        trainer.train()
        # trainer.save_model()
        print(f"{datetime.datetime.now()}: End of experiments for cfg: {config_dict}")


def run_agent(data_fold, yaml_file):
    dataset_path = yaml_file['DATASET_PATH']

    lm_name = yaml_file['LM_NAME']
    special_tokens = yaml_file['SPECIAL_TOKENS']
    tokenize_in_batch = yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = yaml_file['NO_CUDA']

    template_prompt = yaml_file['TEMPLATE_PROMPT']
    map_labels = yaml_file['MAP_LABELS']
    to_freeze_layers = yaml_file['TO_FREEZE_LAYERS']
    unfreeze_last_n = yaml_file['UNFREEZE_LAST_N']

    print("Tuning params read from yaml file")
    # print(f"{datetime.datetime.now()}: BEGIN OF THE EXPERIMENTS")

    # load the dataset
    df_trainset, df_valset = load_dataset(f"{dataset_path}/fold_{data_fold}/")
    print(f"# of samples for training:{len(df_trainset)}")
    print(f"# of samples for validation:{len(df_valset)}")

    tokenizer, lm, lm_config_class = load_language_model_objects(lm_name, special_tokens)
    print("Downloaded tokenizer, model and cfg!")

    # TODO prepare dataset with prepare_train()
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

    # TODO train with train()
    out_dir = yaml_file['OUT_DIR']
    train(out_dir, lm, tokenized_train, tokenized_val, no_cuda)


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
        help="The API key of wandb used to login."
    )

    parser.add_argument(
        "--sweep_id",
        default=None,
        type=str,
        required=True,
        help="The id of the sweep."
    )

    args = parser.parse_args()

    # read params from yaml file
    setting_yaml_file = open(f"{args.setting_path}{args.setting_name}")
    parsed_yaml_file = yaml.load(setting_yaml_file, Loader=yaml.FullLoader)

    fold = parsed_yaml_file['FOLD']
    n_sweep_runs = parsed_yaml_file['N_SWEEP_RUNS']

    print(f"{datetime.datetime.now()}: Begin of the experiments for fold:{fold}")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    sweep_id = f"cdiego89/counterfactual-generation/{args.sweep_id}"
    print(f"Sweep id:{sweep_id}")

    try:
        wandb.agent(sweep_id, function=lambda: run_agent(fold, parsed_yaml_file), count=n_sweep_runs)

    except wandb.errors.CommError:
        print(f"wandb.errors.CommError: could not find sweep: {sweep_id}")
        sys.exit()

    print(f"{datetime.datetime.now()}: End of experiments for fold:{fold}")
    sys.exit()


if __name__ == "__main__":
    main()
