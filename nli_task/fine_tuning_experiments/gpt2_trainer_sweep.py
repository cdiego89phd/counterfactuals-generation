import argparse
import datetime
import yaml
import wandb
import sys
import utils
import cad_fine_tuning_trainer


def run_agent(args, data_fold, wandb_project, yaml_file):
    dataset_path = yaml_file['DATASET_PATH']

    lm_name = yaml_file['LM_NAME']
    special_tokens = yaml_file['SPECIAL_TOKENS']
    tokenize_in_batch = yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = yaml_file['NO_CUDA']

    template_prompt = yaml_file['TEMPLATE_PROMPT']
    out_name = yaml_file['OUT_DIR']
    print("Tuning params read from yaml file")

    # load the dataset
    df_trainset, df_valset, _ = utils.load_dataset(f"{dataset_path}/fold_{data_fold}/")
    if args.debug_mode:
        df_trainset = df_trainset[:10]
        df_valset = df_valset[:10]
    print(f"# of samples for training:{len(df_trainset)}")
    print(f"# of samples for validation:{len(df_valset)}")

    tokenizer, _, _ = utils.load_gpt2_objects(yaml_file['BASE_MODEL'], special_tokens)

    # load the language model
    if yaml_file['MODEL_FROM_LOCAL']:
        model_local_path = f"{yaml_file['MODEL_DIR']}/{lm_name}"
        lm = utils.load_gpt2_from_local(model_local_path)

        # add new, random embeddings for the new tokens
        # this might be needed if the model has been pre-trained with a different tokenizer (of different lenght)
        lm.resize_token_embeddings(len(tokenizer))
    else:
        _, lm, _ = utils.load_gpt2_objects(lm_name, special_tokens)

    print("Downloaded tokenizer, model and cfg!")

    # wrap the datasets with the prompt template
    df_trainset["wrapped_input"] = df_trainset.apply(lambda row: utils.wrap_nli_dataset_with_prompt(row,
                                                                                                    template_prompt,
                                                                                                    special_tokens),
                                                     axis=1)
    print("Training set wrapped!")
    df_valset["wrapped_input"] = df_valset.apply(lambda row: utils.wrap_nli_dataset_with_prompt(row,
                                                                                                template_prompt,
                                                                                                special_tokens),
                                                 axis=1)
    print("Validation set wrapped!")

    tokenized_train, tokenized_val = cad_fine_tuning_trainer.prepare_training(df_trainset,
                                                                              df_valset,
                                                                              tokenizer,
                                                                              tokenize_in_batch)
    print("Datasets have been tokenized successfully!")

    training_cfgs = None  # because this is a sweep agent
    cad_fine_tuning_trainer.train(out_name, lm, tokenized_train, tokenized_val,
                                  no_cuda, training_cfgs, wandb_project, None, False)


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
        "--sweep_id",
        default=None,
        type=str,
        required=True,
        help="The id of the sweep."
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
    n_sweep_runs = parsed_yaml_file['N_SWEEP_RUNS']

    print(f"{datetime.datetime.now()}: Begin of the experiments for fold:{fold}")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    sweep_id = f"{args.wandb_project}/{args.sweep_id}"
    print(f"Sweep id:{sweep_id}")

    try:
        wandb.agent(sweep_id, function=lambda: run_agent(args, fold, args.wandb_project,
                                                         parsed_yaml_file), count=n_sweep_runs)

    except wandb.errors.CommError:
        print(f"wandb.errors.CommError: could not find sweep: {sweep_id}")
        sys.exit()

    print(f"{datetime.datetime.now()}: End of experiments for fold:{fold}")
    sys.exit()


if __name__ == "__main__":
    main()
