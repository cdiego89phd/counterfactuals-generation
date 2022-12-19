import pandas as pd
import numpy as np
import argparse
import datetime
import yaml
import torch
from sentiment_task import generator, utils


# TODO
# Generate $m$ corresponding counterfactuals from $D_{seed}$. We use CouRGe*(CouRGe with GPT2 fine-tuned on SFT*).
# We produce only one counterfactual per data point, and at the end of the generation, we check the LFS, LEV and TED
# (just to check the quality of the new points).


def append_prompt(parsed_yaml_file, gen_testset, n_to_generate) -> pd.DataFrame:
    generation_prompt = parsed_yaml_file['GENERATION_PROMPT']

    # parse the prompt
    generation_prompt = generation_prompt.replace("<", "")
    generation_prompt = generation_prompt.replace(">", "")
    parsed_prompt = generation_prompt.split(" ")
    indexes = [eval(i) for i in parsed_prompt]

    # append the words to the counterfactual
    for idx in range(n_to_generate):
        gen_testset[f"generated_counter_{idx}"] = gen_testset.apply(
            lambda row: append_to_counter(row, indexes, idx), axis=1)

    return gen_testset


def append_to_counter(row, idxs, idx) -> str:
    example = row['example'].split(" ")
    to_add = ""
    for i in idxs:
        to_add += example[i] + " "
    to_return = to_add + row[f"generated_counter_{idx}"]

    return to_return


def main():

    # read params from command line
    parser = argparse.ArgumentParser()
    # SETTINGS_PATH = "/home/diego/counterfactuals-generation/cata_sentiment/settings/"
    parser.add_argument(
        "--setting_path",
        default=None,
        type=str,
        required=True,
        help="The absolute path of the file settings."
    )

    # e.g. SETTING_NAME = "generation_prompt-1.yaml"
    parser.add_argument(
        "--setting_name",
        default=None,
        type=str,
        required=True,
        help="The name of yaml file where to load the setting from."
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

    dataset_path = parsed_yaml_file['DATASET_PATH']
    lm_name = parsed_yaml_file['LM_NAME']
    base_lm_name = parsed_yaml_file['BASE_MODEL']
    special_tokens = parsed_yaml_file['SPECIAL_TOKENS']
    n_to_generate = parsed_yaml_file['N_TO_GENERATE']
    gen_params = parsed_yaml_file['GEN_CFGS']

    # set Random seed
    torch.manual_seed(parsed_yaml_file["SEED"])
    np.random.seed(parsed_yaml_file["SEED"])

    print(f"{datetime.datetime.now()}: Begin generation")

    # load the seed dataset
    df_seed = pd.read_csv(f"{dataset_path}seed_data.csv", sep='\t')
    if args.debug_mode:
        df_seed = df_seed[:10]
    print(f"{datetime.datetime.now()}: Seed data loaded")
    print(f"# of samples in seed data:{len(df_seed)}")

    tokenizer, _, _ = utils.load_gpt2_objects(base_lm_name, special_tokens)
    model_local_path = f"{parsed_yaml_file['MODEL_DIR']}/{lm_name}"
    generator_lm = utils.load_gpt2_from_local(model_local_path)
    print(f"{datetime.datetime.now()}: Language model loaded from local")

    # generate counterfactuals from the seed dataset
    gen_data = generator.generate_counterfactuals(parsed_yaml_file,
                                                  df_seed,
                                                  generator_lm,
                                                  tokenizer,
                                                  gen_params,
                                                  n_to_generate)
    df_gen = gen_data.dataframe_from_dataset(n_to_generate)
    df_gen.to_csv(f"{dataset_path}counterfactuals_data.csv", sep='\t', header=True, index=False)
    print(f"{datetime.datetime.now()}: Generation completed!")

    print(f"{datetime.datetime.now()}: Creating CATA data...")
    # load the n_data dataset
    n_data = pd.read_csv(f"{dataset_path}n_data.csv", sep='\t')
    n_data.drop(columns=["sentiment_ex", "review_len", "label_counter", "sentiment_counter"])
    df_gen.rename(columns={"label_counter": "label", "counterfactual": "text"}, inplace=True)

    # produce training dataset
    df_gen.drop(columns=["paired_id", "label_ex", "example"])
    df_gen.rename(columns={"label_counter": "label", "counterfactual": "text"}, inplace=True)

    assert df_gen.columns == n_data.columns
    training_data = pd.concat([n_data, df_gen])

    # print training data
    training_data.to_csv(f"{dataset_path}training_data.csv", sep='\t', header=True, index=False)

    print(f"{datetime.datetime.now()}: End generation!")

    return


if __name__ == "__main__":
    main()

