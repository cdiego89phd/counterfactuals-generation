import argparse
import datetime
import yaml
from sentiment_task.fine_tuning_experiments import cad_fine_tuning_trainer
from sentiment_task import utils


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
    map_labels = parsed_yaml_file['MAP_LABELS']

    print("Training's params read from yaml file")
    print(f"{datetime.datetime.now()}: TUNING BEGINS for fold {fold}")

    # load the dataset
    df_trainset, df_valset, _ = utils.load_dataset(f"{dataset_path}/fold_{fold}/")
    if args.debug_mode:
        df_trainset = df_trainset[:10]
        df_valset = df_valset[:10]
    print(f"# of samples for training:{len(df_trainset)}")
    print(f"# of samples for validation:{len(df_valset)}")

    # load the language model
    tokenizer, lm, lm_config_class = utils.load_gpt2_objects(lm_name, special_tokens)
    print("Downloaded tokenizer, model and cfg!")

    # wrap the datasets with the prompt template
    df_trainset["wrapped_input"] = df_trainset.apply(lambda row: utils.wrap_dataset_with_prompt(row,
                                                                                                template_prompt,
                                                                                                map_labels,
                                                                                                special_tokens), axis=1)
    print("Training set wrapped!")
    df_valset["wrapped_input"] = df_valset.apply(lambda row: utils.wrap_dataset_with_prompt(row,
                                                                                            template_prompt,
                                                                                            map_labels,
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

    run_name = f"{lm_name}@{prompt_id}@cad_fine_tuning"
    out_name = f"{out_dir}/{lm_name}"

    cad_fine_tuning_trainer.train(out_name, lm, tokenized_train, tokenized_val,
                                  no_cuda, training_cfgs, args.wandb_project, run_name)

    print(f"{datetime.datetime.now()}: End of experiments for fold:{fold}")


if __name__ == "__main__":
    main()
