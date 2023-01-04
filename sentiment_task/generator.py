import pandas as pd
import numpy as np
import argparse
import datetime
import yaml
import torch
import openprompt
import generation
import utils

from openprompt.prompts import ManualTemplate
from openprompt.plms.lm import LMTokenizerWrapper

# try:
#     from kernl.model_optimization import optimize_model
#     kernl_imported = True
# except ImportError:
#     kernl_imported = False
#     print("Kernl module not found! GPU optimization not available for inference")


# TODO: this method is used in "interactive_console.py";
# But the implementation is not complete
def generate_single_counterfactual(yaml_file,
                                   prompt,
                                   seed_review,
                                   seed_class,
                                   trained_lm,
                                   tokenizer,
                                   cuda_device=0,
                                   n_to_generate=1) -> str:

    special_tokens = yaml_file['SPECIAL_TOKENS']
    map_labels = yaml_file['MAP_LABELS']
    gen_params = yaml_file['GEN_PARAMS']

    return "None"


def generate_counterfactuals(yaml_file,
                             df_testset,
                             trained_lm,
                             tokenizer,
                             gen_params,
                             n_to_generate=1) -> pd.DataFrame:

    special_tokens = yaml_file['SPECIAL_TOKENS']
    map_labels = yaml_file['MAP_LABELS']
    generation_prompt = yaml_file['GENERATION_PROMPT']

    # wrap the datasets with the prompt template
    df_testset["wrapped_input"] = df_testset.apply(lambda row: utils.wrap_dataset_with_prompt(row,
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
    df_counter = utils.SentimentDataset.to_dataframe(n_to_generate, counter_generator.get_dataset())

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

    parser.add_argument(
        "--run_kernl",
        default=0,
        type=int,
        required=False,
        help="Whether to speed up training with kernl library."
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

    print(f"{datetime.datetime.now()}: Begin GEN TUNING for fold:{fold}")

    # load the dataset
    dataset_path = parsed_yaml_file['DATASET_PATH']
    _, _, df_testset = utils.load_dataset(f"{dataset_path}/fold_{fold}/")
    if args.debug_mode:
        df_testset = df_testset[:10]
    print(f"{datetime.datetime.now()}: Test set loaded for fold:{fold}")
    print(f"# of samples for test:{len(df_testset)}")

    tokenizer, trained_lm, _ = utils.load_gpt2_objects(base_lm_name, special_tokens)
    if parsed_yaml_file['MODEL_FROM_LOCAL']:
        model_local_path = f"{parsed_yaml_file['MODEL_DIR']}/{parsed_yaml_file['LM_NAME']}"
        trained_lm = utils.load_gpt2_from_local(model_local_path)
    print(f"{datetime.datetime.now()}: Language model loaded from local:{parsed_yaml_file['MODEL_FROM_LOCAL']}")

    # if args.run_kernl and kernl_imported:
    #     trained_lm.eval().cuda()
    #     optimize_model(trained_lm)
    #     print("Runnning Kernel optimization!!")

    # generate the counterfactuals
    gen_params = parsed_yaml_file['GEN_CFGS']

    # we generate n_to_generate counterfactuals
    gen_testset = generate_counterfactuals(parsed_yaml_file, df_testset, trained_lm, tokenizer,
                                           gen_params, n_to_generate)
    df_gen_testset = gen_testset.dataframe_from_dataset(n_to_generate)

    print("Generation completed!")

    # print test generation
    gen_filename = f"{lm_name}{parsed_yaml_file['OUT_LABEL']}.csv"
    df_gen_testset.to_csv(f"{parsed_yaml_file['OUT_DIR']}{gen_filename}", sep='\t', header=True, index=False)

    print(f"{datetime.datetime.now()}: End GEN TUNING for fold:{fold}")

    return


if __name__ == "__main__":
    main()
