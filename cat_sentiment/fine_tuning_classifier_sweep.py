import argparse
import datetime
import yaml
import wandb
import sys
import os
import transformers
import utils
from cat_sentiment import fine_tuning_classifier


def run_agent(args, wandb_project, yaml_file):
    val_prop = yaml_file['VAL_PROP']
    lm_name = yaml_file['LM_NAME']
    tokenize_in_batch = yaml_file['TOKENIZE_IN_BATCH']
    no_cuda = yaml_file['NO_CUDA']

    random_seed = yaml_file['RANDOM_SEED']
    print("Tuning params read from yaml file")

    # load the dataset
    df_trainset, df_valset = utils.load_dataset_with_val(random_seed,
                                                         val_prop,
                                                         f"{args.dataset_path}/{args.dataset_name}.csv"
                                                         )

    df_trainset = fine_tuning_classifier.clean_dataset(df_trainset, "trainset")
    df_valset = fine_tuning_classifier.clean_dataset(df_valset, "trainset")

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

    tokenized_train, tokenized_val = fine_tuning_classifier.prepare_training(df_trainset,
                                                                             df_valset,
                                                                             tokenizer,
                                                                             tokenize_in_batch)
    print("Datasets have been tokenized successfully!")

    fine_tuning_classifier.train(args.out_dir,
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
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="The path of the dir to use to save sweep files."
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

    print("parsed YAML file!")

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
        wandb.finish()
        sys.exit()

    print(f"{datetime.datetime.now()}: End of experiments for dataset:{args.dataset_name}")
    wandb.finish()
    sys.exit()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
