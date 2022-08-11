import pandas as pd
import argparse
import datetime
import yaml
import openprompt
from sentiment_task import generation, utils

from openprompt.prompts import ManualTemplate
from openprompt.plms.lm import LMTokenizerWrapper


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
                             n_to_generate=1) -> generation.CounterGenerator:

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
        tokenizer_wrapper_class=tokenizer_wrapper
    )

    counter_generator = generation.CounterGenerator(prompt_template,
                                                    trained_lm,
                                                    test_data_loader,
                                                    test_set,
                                                    gen_params)

    counter_generator.perform_generation(tokenizer, n_to_generate)

    # the generated counterfactuals are held inside the counter_generator object
    return counter_generator


def dataframe_from_dataset(gen_valset) -> pd.DataFrame:
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


def main():

    # read params from command line
    parser = argparse.ArgumentParser()
    # SETTINGS_PATH = "/home/diego/counterfactuals-generation/sentiment_task/zero_shot_experiments/settings/"
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
    # cuda_device = parsed_yaml_file['CUDA_DEVICE']
    # n_to_generate = parsed_yaml_file['N_TO_GENERATE']

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

    # generate the counterfactuals
    gen_params = parsed_yaml_file['GEN_CFGS']
    gen_testset = generate_counterfactuals(parsed_yaml_file, df_testset, trained_lm, tokenizer,
                                           gen_params)
    df_gen_testset = gen_testset.dataframe_from_dataset()
    print("Generation completed!")

    # print test generation
    prompt_id = parsed_yaml_file['PROMPT_ID']
    if parsed_yaml_file['MODEL_FROM_LOCAL']:
        gen_filename = f"{lm_name}.csv"
    else:
        gen_filename = f"{lm_name}.csv"
    df_gen_testset.to_csv(f"{parsed_yaml_file['OUT_DIR']}{gen_filename}", sep='\t', header=True, index=False)

    print(f"{datetime.datetime.now()}: End GEN TUNING for fold:{fold}")


if __name__ == "__main__":
    main()
