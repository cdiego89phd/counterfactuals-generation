import pandas as pd
import transformers
import datasets
import argparse
import datetime
import yaml
import wandb
import sys
import openprompt
from sentiment_task import evaluation
from sentiment_task import generation

from openprompt.prompts import ManualTemplate
from openprompt.plms.lm import LMTokenizerWrapper
import os


# TODO remove the filter for debug
def load_dataset(loading_path):
    trainset = pd.read_csv(loading_path + "training_set", sep='\t')
    val = pd.read_csv(loading_path + "val_set", sep='\t')
    return trainset, val
    # return trainset[:100], val[:100]


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


def train(out_dir, model, trainset, valset, no_cuda, train_cfg):

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        no_cuda=no_cuda,
        num_train_epochs=train_cfg["EPOCHS"],
        per_device_train_batch_size=train_cfg["TRAIN_BATCHSIZE"],
        per_device_eval_batch_size=train_cfg["EVAL_BATCHSIZE"],
        gradient_accumulation_steps=train_cfg["BATCH_UPDATE"],
        do_eval=False,
        evaluation_strategy=transformers.IntervalStrategy.EPOCH,
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
        eval_dataset=valset
    )

    trainer.train()
    return trainer.model


def train_model(yaml_file):

    fold = yaml_file['FOLD']
    dataset_path = yaml_file['DATASET_PATH']

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

    out_dir = yaml_file['OUT_DIR']
    trained_lm = train(out_dir, lm, tokenized_train, tokenized_val, no_cuda, train_cfg)

    return trained_lm, tokenizer


def prepare_classifier(classifier_name):
    # load the sentiment classifier
    classifier_tokenizer = transformers.AutoTokenizer.from_pretrained(classifier_name)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_name)
    classifier_label_map = {'NEGATIVE':  0, 'POSITIVE': 1}
    classification_tools = {"tokenizer": classifier_tokenizer,
                            "classifier": classifier,
                            "label_map": classifier_label_map}

    return classification_tools


def generate_counterfactuals(yaml_file, df_valset, trained_lm, tokenizer, gen_params):

    special_tokens = yaml_file['SPECIAL_TOKENS']
    map_labels = yaml_file['MAP_LABELS']
    generation_prompt = yaml_file['GENERATION_PROMPT']

    # wrap the datasets with the prompt template
    df_valset["wrapped_input"] = df_valset.apply(lambda row: wrap_with_prompt(row,
                                                                              generation_prompt,
                                                                              map_labels,
                                                                              special_tokens), axis=1)

    # prepare the data loader
    valset = generation.SentimentDataset(raw_dataframe=df_valset.copy(deep=True))
    valset.prepare_dataloader()

    template_prompt = '{"placeholder":"text_a"}{"mask"}'
    prompt_template = ManualTemplate(text=template_prompt, tokenizer=tokenizer)
    tokenizer_wrapper = LMTokenizerWrapper
    val_data_loader = openprompt.PromptDataLoader(
        dataset=list(valset.get_dataset().values()),
        tokenizer=tokenizer,
        template=prompt_template,
        tokenizer_wrapper_class=tokenizer_wrapper
    )

    counter_generator = generation.CounterGenerator(prompt_template,
                                                    trained_lm,
                                                    val_data_loader,
                                                    valset,
                                                    gen_params)

    counter_generator.perform_generation(not yaml_file['NO_CUDA'], tokenizer)

    # the generated counterfactuals are held inside the counter_generator object
    return counter_generator.dataset


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


def run_agent(yaml_file, trained_lm, tokenizer, classification_tools):

    fold = yaml_file['FOLD']
    dataset_path = yaml_file['DATASET_PATH']

    # load the dataset (we only use the valset)
    _, df_valset = load_dataset(f"{dataset_path}/fold_{fold}/")

    with wandb.init(settings=wandb.Settings(console='off'),
                    project="conterfactuals-generation"):
        gen_params = wandb.config

        gen_valset = generate_counterfactuals(yaml_file, df_valset, trained_lm, tokenizer, gen_params)
        print("Generation completed!")

        eval_valset = dataframe_from_dataset(gen_valset)
        evaluator = evaluation.SentimentEvaluator(classification_tools["tokenizer"],
                                                  classification_tools["classifier"],
                                                  classification_tools["label_map"],
                                                  trained_lm.device.index)

        eval_valset = evaluator.clean_evalset(eval_valset)
        evaluator.infer_predictions(eval_valset)
        lf_score = evaluator.calculate_lf_score(eval_valset)
        conf_score = evaluator.get_conf_score_pred()
        blue_mean, blue_var = evaluator.calculate_blue_score(eval_valset)

        wandb.log({"lf_score": lf_score,
                   "conf_score": conf_score,
                   "blue_mean": blue_mean,
                   "blue_var": blue_var})


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
    classifier_name = parsed_yaml_file['CLASSIFIER_NAME']

    print(f"{datetime.datetime.now()}: Begin GEN TUNING for fold:{fold}")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    sweep_id = f"cdiego89/counterfactuals-generation/{args.sweep_id}"
    print(f"Sweep id:{sweep_id}")

    trained_lm, tokenizer = train_model(parsed_yaml_file)
    classification_tools = prepare_classifier(classifier_name)

    try:
        wandb.agent(sweep_id, function=lambda: run_agent(parsed_yaml_file, trained_lm, tokenizer, classification_tools),
                    count=n_sweep_runs)

    except wandb.errors.CommError:
        print(f"wandb.errors.CommError: could not find sweep: {sweep_id}")
        sys.exit()

    print(f"{datetime.datetime.now()}: End GEN TUNING for fold:{fold}")
    sys.exit()


if __name__ == "__main__":
    os.environ['WANDB_CONSOLE'] = 'off'  # this will prevent the sweep to finish with no errors
    main()
