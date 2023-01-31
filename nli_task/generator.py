import numpy as np
import argparse
import datetime

import pandas as pd
import yaml
import torch
import utils
import generation
import openprompt

from openprompt.prompts import ManualTemplate
from openprompt.plms.lm import LMTokenizerWrapper


def generate_counterfactuals(yaml_file,
                             df_testset,
                             trained_lm,
                             tokenizer,
                             gen_params,
                             n_to_generate=1) -> pd.DataFrame:

    special_tokens = yaml_file['SPECIAL_TOKENS']
    generation_prompt = yaml_file['GENERATION_PROMPT']

    # wrap the datasets with the prompt template
    df_testset["wrapped_input"] = df_testset.apply(lambda row: utils.wrap_nli_dataset_with_prompt(row,
                                                                                                  generation_prompt,
                                                                                                  special_tokens),
                                                   axis=1)

    # prepare the data loader
    test_set = utils.NLIDataset(raw_dataframe=df_testset.copy(deep=True))
    test_set.prepare_dataloader()

    template_prompt = '{"placeholder":"text_a"}{"mask"}'
    prompt_template = ManualTemplate(text=template_prompt, tokenizer=tokenizer)
    tokenizer_wrapper = LMTokenizerWrapper
    test_data_loader = openprompt.PromptDataLoader(
        dataset=list(test_set.get_dataset().values()),
        tokenizer=tokenizer,
        template=prompt_template,
        tokenizer_wrapper_class=tokenizer_wrapper,
        # truncate_method="tail",
    )

    counter_generator = generation.CounterGenerator(prompt_template,
                                                    trained_lm,
                                                    test_data_loader,
                                                    test_set,
                                                    gen_params)

    # the generated counterfactuals are held inside the counter_generator object
    counter_generator.perform_generation(tokenizer, n_to_generate)

    # make a dataframe
    df_counter = utils.NLIDataset.to_dataframe(n_to_generate, counter_generator.get_dataset())

    return df_counter


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

    fold = parsed_yaml_file['FOLD']
    lm_name = parsed_yaml_file['LM_NAME']
    base_lm_name = parsed_yaml_file['BASE_LM_NAME']
    special_tokens = parsed_yaml_file['SPECIAL_TOKENS']
    n_to_generate = parsed_yaml_file['N_TO_GENERATE']

    # set Random seed
    torch.manual_seed(parsed_yaml_file["SEED"])
    np.random.seed(parsed_yaml_file["SEED"])

    print(f"{datetime.datetime.now()}: Begin GEN for fold:{fold}")

    # load the dataset
    dataset_path = parsed_yaml_file['DATASET_PATH']
    _, _, df_testset = utils.load_dataset(f"{dataset_path}/fold_{fold}/")
    if args.debug_mode:
        df_testset = df_testset[:10]
    print(f"{datetime.datetime.now()}: Test set loaded for fold:{fold}")
    print(f"# of samples for test:{len(df_testset)}")

    # tokenizer = utils.load_tokenizer(base_lm_name, special_tokens)
    tokenizer = utils.load_tokenizer_bis(base_lm_name, special_tokens)

    if parsed_yaml_file['MODEL_FROM_LOCAL']:
        model_local_path = f"{parsed_yaml_file['MODEL_DIR']}/{parsed_yaml_file['LM_NAME']}"
        trained_lm = utils.load_causal_model_from_local(model_local_path)
        # trained_lm.resize_token_embeddings(len(tokenizer))
    else:
        trained_lm, _ = utils.load_causal_model(base_lm_name, len(tokenizer), special_tokens)
    print(f"{datetime.datetime.now()}: Language model loaded from local:{parsed_yaml_file['MODEL_FROM_LOCAL']}")

    # generate the counterfactuals
    gen_params = parsed_yaml_file['GEN_CFGS']

    # we generate n_to_generate counterfactuals
    # we return a dataframe
    df_gen_testset = generate_counterfactuals(parsed_yaml_file,
                                              df_testset,
                                              trained_lm,
                                              tokenizer,
                                              gen_params,
                                              n_to_generate)

    print("Generation completed!")

    # print test generation
    gen_filename = f"{lm_name}{parsed_yaml_file['OUT_LABEL']}.csv"
    df_gen_testset.to_csv(f"{parsed_yaml_file['OUT_DIR']}{gen_filename}", sep='\t', header=True, index=False)

    print(f"{datetime.datetime.now()}: End GEN for fold:{fold}")

    return


if __name__ == "__main__":
    main()
