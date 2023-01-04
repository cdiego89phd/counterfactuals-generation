import pandas as pd
import argparse
import datetime
import yaml
import utils
from sentiment_task import generator


def main():
    # read params from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        required=True,
        help="The absolute path of the dataset."
    )

    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        help="The name of the dataset."
    )

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
        "--output_name",
        default=None,
        type=str,
        required=True,
        help="The name of the file output."
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

    base_lm_name = parsed_yaml_file['BASE_LM_NAME']
    special_tokens = parsed_yaml_file['SPECIAL_TOKENS']
    n_to_generate = parsed_yaml_file['N_TO_GENERATE']

    # load dataset
    df_testset = pd.read_csv(f"{args.dataset_path}/{args.dataset_name}", sep='\t')
    if args.debug_mode:
        df_testset = df_testset[:12]
    print(f"{datetime.datetime.now()}: Test set loaded")
    print(f"# of samples for test:{len(df_testset)}")

    tokenizer, _, _ = utils.load_gpt2_objects(base_lm_name, special_tokens)
    model_local_path = f"{parsed_yaml_file['MODEL_DIR']}/{parsed_yaml_file['LM_NAME']}"
    trained_lm = utils.load_gpt2_from_local(model_local_path)
    print(f"{datetime.datetime.now()}: Language model loaded from local.")

    # generate the counterfactuals
    gen_params = parsed_yaml_file['GEN_CFGS']

    print(f"{datetime.datetime.now()}: Begin GEN TUNING")
    # we generate n_to_generate counterfactuals
    df_gen_testset = generator.generate_counterfactuals(parsed_yaml_file,
                                                        df_testset,
                                                        trained_lm,
                                                        tokenizer,
                                                        gen_params,
                                                        n_to_generate)

    # print test generation
    gen_filename = f"{args.output_name}.csv"
    df_gen_testset.to_csv(f"{parsed_yaml_file['OUT_DIR']}{gen_filename}", sep='\t', header=True, index=False)

    print(f"{datetime.datetime.now()}: End GEN TUNING")


if __name__ == "__main__":
    main()
