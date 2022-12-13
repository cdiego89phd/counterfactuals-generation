import argparse
import datetime
import yaml
import wandb
import sys
import datasets
import transformers
from sentiment_task import utils
from cata_sentiment import fine_tuning_classifier


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


def run_agent(args, wandb_project, yaml_file):
    val_prop = yaml_file['VAL_PROP']
    out_dir = yaml_file['OUT_DIR']

    lm_name = yaml_file['LM_NAME']
    tokenize_in_batch = yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = yaml_file['NO_CUDA']

    # map_labels = yaml_file['MAP_LABELS']
    random_seed = yaml_file['RANDOM_SEED']
    # out_name = yaml_file['OUT_DIR']
    print("Tuning params read from yaml file")

    # load the dataset
    df_trainset, df_valset = utils.load_dataset_with_val(random_seed,
                                                         val_prop,
                                                         f"{args.dataset_path}/{args.dataset_name}.csv"
                                                         )
    if args.debug_mode:
        df_trainset = df_trainset[:10]
        df_valset = df_valset[:10]
    print(f"# of samples for training:{len(df_trainset)}")
    print(f"# of samples for validation:{len(df_valset)}")

    # load the language model
    # TODO check on the lenght of tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(lm_name)
    lm = transformers.AutoModelForSequenceClassification.from_pretrained(lm_name)

    print("Downloaded tokenizer, model and cfg!")

    tokenized_train, tokenized_val = prepare_training(df_trainset,
                                                      df_valset,
                                                      tokenizer,
                                                      tokenize_in_batch)
    print("Datasets have been tokenized successfully!")

    fine_tuning_classifier.train(out_dir,
                                 lm,
                                 tokenized_train,
                                 tokenized_val,
                                 no_cuda,
                                 None,  # training cfgs
                                 wandb_project,
                                 None,  # run_name
                                 False  # save_model
                                 )


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

    n_sweep_runs = parsed_yaml_file['N_SWEEP_RUNS']

    print(f"{datetime.datetime.now()}: Begin of the experiments for dataset:{args.dataset_name}")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    sweep_id = f"{args.wandb_project}/{args.sweep_id}"
    print(f"Sweep id:{sweep_id}")

    try:
        wandb.agent(sweep_id, function=lambda: run_agent(args,
                                                         args.wandb_project,
                                                         parsed_yaml_file), count=n_sweep_runs)

    except wandb.errors.CommError:
        print(f"wandb.errors.CommError: could not find sweep: {sweep_id}")
        sys.exit()

    print(f"{datetime.datetime.now()}: End of experiments for dataset:{args.dataset_name}")
    sys.exit()


if __name__ == "__main__":
    main()
