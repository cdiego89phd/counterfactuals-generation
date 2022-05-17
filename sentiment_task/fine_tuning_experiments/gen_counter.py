import pandas as pd
import transformers
import datasets
import argparse
import datetime
import yaml
import openprompt
from sentiment_task import generation

from openprompt.prompts import ManualTemplate
from openprompt.plms.lm import LMTokenizerWrapper


# TODO remove the filter for debug
def load_dataset(loading_path):
    trainset = pd.read_csv(loading_path + "training_set", sep='\t')
    val = pd.read_csv(loading_path + "val_set", sep='\t')
    testset = pd.read_csv(loading_path + "test_set", sep='\t')
    return trainset, val, testset
    # return trainset[:10], val[:10], testset[:10]


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


def train(out_dir, model, trainset, no_cuda, train_cfg):

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        no_cuda=no_cuda,
        num_train_epochs=train_cfg["EPOCHS"],
        per_device_train_batch_size=train_cfg["TRAIN_BATCHSIZE"],
        per_device_eval_batch_size=train_cfg["EVAL_BATCHSIZE"],
        gradient_accumulation_steps=train_cfg["BATCH_UPDATE"],
        do_eval=False,
        evaluation_strategy='no',
        warmup_steps=train_cfg["WARMUP_STEPS"],
        learning_rate=train_cfg["LR"],
        adam_epsilon=train_cfg["EPS"],
        weight_decay=train_cfg["WEIGHT_DECAY"],
        save_total_limit=0,
        save_strategy='no',
        load_best_model_at_end=False,
        report_to='none'
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        # eval_dataset=trainset
    )

    trainer.train()
    return trainer.model


def train_model(yaml_file, df_trainset, df_valset):

    lm_name = yaml_file['LM_NAME']
    special_tokens = yaml_file['SPECIAL_TOKENS']
    tokenize_in_batch = yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = yaml_file['NO_CUDA']

    template_prompt = yaml_file['TEMPLATE_PROMPT']
    map_labels = yaml_file['MAP_LABELS']
    to_freeze_layers = yaml_file['TO_FREEZE_LAYERS']
    unfreeze_last_n = yaml_file['UNFREEZE_LAST_N']

    train_cfg = yaml_file["TRAIN_CFG"]

    print("Tuning params read from yaml file")

    tokenizer, lm, lm_config_class = load_language_model_objects(lm_name, special_tokens)
    print("Downloaded tokenizer, model and cfg!")

    df_train = pd.concat([df_trainset, df_valset], ignore_index=True, copy=True)
    print(f"# of total samples for training (train + val):{len(df_train)}")

    # wrap the datasets with the prompt template
    df_train["wrapped_input"] = df_train.apply(lambda row: wrap_with_prompt(row,
                                                                            template_prompt,
                                                                            map_labels,
                                                                            special_tokens), axis=1)
    print("Training set wrapped!")

    # convert dataset from pandas to Dataset
    training_set = datasets.Dataset.from_pandas(df_train)

    # TOKENIZE datasets
    tokenized_train = training_set.map(lambda examples: tokenizer(examples["wrapped_input"],
                                                                  padding="max_length",
                                                                  truncation=True), batched=tokenize_in_batch)
    tokenized_train = tokenized_train.add_column("labels", tokenized_train['input_ids'])
    print("Training set has been tokenized successfully!")

    lm = freeze_layers_lm(to_freeze_layers, unfreeze_last_n, lm)

    out_dir = yaml_file['OUT_DIR']
    trained_lm = train(out_dir, lm, tokenized_train, no_cuda, train_cfg)

    return trained_lm, tokenizer


def generate_counterfactuals(yaml_file, df_testset, trained_lm, tokenizer, gen_params):

    special_tokens = yaml_file['SPECIAL_TOKENS']
    map_labels = yaml_file['MAP_LABELS']
    generation_prompt = yaml_file['GENERATION_PROMPT']

    # wrap the datasets with the prompt template
    df_testset["wrapped_input"] = df_testset.apply(lambda row: wrap_with_prompt(row,
                                                                                generation_prompt,
                                                                                map_labels,
                                                                                special_tokens), axis=1)

    # prepare the data loader
    test_set = generation.SentimentDataset(raw_dataframe=df_testset.copy(deep=True))
    test_set.prepare_dataloader()

    template_prompt = '{"placeholder":"text_a"}{"mask"}'
    prompt_template = ManualTemplate(text=template_prompt, tokenizer=tokenizer)
    tokenizer_wrapper = LMTokenizerWrapper
    test_data_loader = openprompt.PromptDataLoader(
        dataset=list(test_set.get_dataset().values()),
        tokenizer=tokenizer,
        template=prompt_template,
        tokenizer_wrapper_class=tokenizer_wrapper
    )

    counter_generator = generation.CounterGenerator(prompt_template,
                                                    trained_lm,
                                                    test_data_loader,
                                                    test_set,
                                                    gen_params)

    counter_generator.perform_generation(not yaml_file['NO_CUDA'], tokenizer)

    # the generated counterfactuals are held inside the counter_generator object
    return counter_generator


def dataframe_from_dataset(gen_valset):
    """Build a dataframe from dataset"""

    paired_ids = [idx for idx in gen_valset]
    labels_ex = [gen_valset.__getitem__(idx).meta["label_ex"] for idx in gen_valset]
    examples = [gen_valset.__getitem__(idx).meta["example"] for idx in gen_valset]
    labels_counter = [gen_valset.__getitem__(idx).meta["label_counter"] for idx in gen_valset]
    counterfactuals = [gen_valset.__getitem__(idx).meta["counterfactual"] for idx in gen_valset]
    generated_counters = [gen_valset.__getitem__(idx).meta["generated_counter"] for idx in gen_valset]
    d = {"paired_id": paired_ids,
         "label_ex": labels_ex,
         "example": examples,
         "label_counter": labels_counter,
         "counterfactual": counterfactuals,
         "generated_counter": generated_counters
         }
    return pd.DataFrame(data=d)


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

    # e.g. SETTING_NAME = "tuning_gen_hypers_prompt-1_fold-0.yaml"
    parser.add_argument(
        "--setting_name",
        default=None,
        type=str,
        required=True,
        help="The name of yaml file where to load the setting from."
    )

    args = parser.parse_args()

    # read params from yaml file
    setting_yaml_file = open(f"{args.setting_path}{args.setting_name}")
    parsed_yaml_file = yaml.load(setting_yaml_file, Loader=yaml.FullLoader)

    folds = parsed_yaml_file['FOLDS']

    for fold in folds:

        print(f"{datetime.datetime.now()}: Begin GEN TUNING for fold:{fold}")

        # load the dataset
        dataset_path = parsed_yaml_file['DATASET_PATH']
        df_trainset, df_valset, df_testset = load_dataset(f"{dataset_path}/fold_{fold}/")
        print(f"# of samples for training:{len(df_trainset)}")
        print(f"# of samples for validation:{len(df_valset)}")
        print(f"# of samples for test:{len(df_testset)}")

        print(f"{datetime.datetime.now()}: Fine tuning language model for fold:{fold}")
        trained_lm, tokenizer = train_model(parsed_yaml_file,
                                            df_trainset,
                                            df_valset)  # fine tune language model

        # generate the counterfactuals
        gen_params = parsed_yaml_file['GEN_CFG']
        gen_testset = generate_counterfactuals(parsed_yaml_file, df_testset, trained_lm, tokenizer, gen_params)
        df_gen_testset = gen_testset.dataframe_from_dataset()
        print("Generation completed!")

        # print test generation
        prompt_id = parsed_yaml_file['PROMPT_ID']
        gen_filename = f"prompt-{prompt_id}_fold-{fold}.csv"
        df_gen_testset.to_csv(f"{parsed_yaml_file['GEN_PATH']}{gen_filename}", sep='\t', header=True, index=False)

        print(f"{datetime.datetime.now()}: End GEN TUNING for fold:{fold}")


if __name__ == "__main__":
    main()
