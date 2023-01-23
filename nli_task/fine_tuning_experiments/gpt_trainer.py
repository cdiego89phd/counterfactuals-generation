import argparse
import datetime
import yaml
import cad_fine_tuning_trainer
import utils
import os


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

    # e.g. SETTING_NAME = "tuning_cad_prompt_1.yaml"
    parser.add_argument(
        "--setting_name",
        default=None,
        type=str,
        required=True,
        help="The name of yaml file where to load the setting from."
    )

    # e.g. SETTING_NAME = "tuning_cad_prompt_1.yaml"
    parser.add_argument(
        "--save_model",
        default=True,
        type=int,
        required=False,
        help="Whether to save the model on a dir."
    )

    parser.add_argument(
        "--wandb_key",
        default=None,
        type=str,
        required=True,
        help="The API key of wandb."
    )

    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        required=True,
        help="The name of the wandb project."
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

    fold = parsed_yaml_file['FOLD']
    out_dir = parsed_yaml_file['OUT_DIR']

    dataset_path = parsed_yaml_file['DATASET_PATH']

    prompt_id = parsed_yaml_file['PROMPT_ID']
    lm_name = parsed_yaml_file['LM_NAME']
    special_tokens = parsed_yaml_file['SPECIAL_TOKENS']
    tokenize_in_batch = parsed_yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = parsed_yaml_file['NO_CUDA']

    template_prompt = parsed_yaml_file['TEMPLATE_PROMPT']

    print("Training's params read from yaml file")
    print(f"{datetime.datetime.now()}: TUNING BEGINS for fold {fold}")

    # load the dataset
    df_trainset, df_valset, _ = utils.load_dataset(f"{dataset_path}/fold_{fold}/")
    if args.debug_mode:
        df_trainset = df_trainset[:100]
        df_valset = df_valset[:100]
    print(f"# of samples for training:{len(df_trainset)}")
    print(f"# of samples for validation:{len(df_valset)}")

    base_name = parsed_yaml_file['BASE_MODEL']
    tokenizer = utils.load_tokenizer(base_name, special_tokens)

    # load the language model
    if parsed_yaml_file['MODEL_FROM_LOCAL']:
        model_local_path = f"{parsed_yaml_file['MODEL_DIR']}/{lm_name}"
        lm, _ = utils.load_causal_model_from_local(model_local_path)

        # add new, random embeddings for the new tokens
        # this might be needed if the model has been pre-trained with a different tokenizer (of different lenght)
        lm.resize_token_embeddings(len(tokenizer))
    else:
        lm, _ = utils.load_causal_model(lm_name, len(tokenizer), special_tokens)

    print("Downloaded tokenizer, model and cfg!")

    # wrap the datasets with the prompt template
    df_trainset["wrapped_input"] = df_trainset.apply(lambda row: utils.wrap_nli_dataset_with_prompt(row,
                                                                                                    template_prompt,
                                                                                                    special_tokens),
                                                     axis=1)
    print("Training set wrapped!")
    df_valset["wrapped_input"] = df_valset.apply(lambda row: utils.wrap_nli_dataset_with_prompt(row,
                                                                                                template_prompt,
                                                                                                special_tokens), axis=1)
    print("Validation set wrapped!")

    tokenized_train, tokenized_val = cad_fine_tuning_trainer.prepare_training(df_trainset,
                                                                              df_valset,
                                                                              tokenizer,
                                                                              tokenize_in_batch)
    print("Datasets have been tokenized successfully!")

    training_cfgs = None
    if not parsed_yaml_file['IS_SWEEP']:
        training_cfgs = parsed_yaml_file['TRAINING_CFGS']

        if "EleutherAI" in lm_name:  # load gptj
            print("Running GPTJ configuration!")
            # training_cfgs["tf32"] = False
            # training_cfgs["fp16"] = True
            training_cfgs["optim"] = "adafactor"
        else:
            # training_cfgs["tf32"] = True
            # training_cfgs["fp16"] = False
            training_cfgs["optim"] = "adamw_hf"

    run_name = f"{lm_name}@prompt-{prompt_id}@cad_fine_tuning"
    out_name = f"{out_dir}/{run_name}"

    cad_fine_tuning_trainer.train(out_name, lm, tokenized_train, tokenized_val,
                                  no_cuda, training_cfgs, args.wandb_project, run_name, args.save_model)

    print(f"{datetime.datetime.now()}: End of experiments for fold:{fold}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
