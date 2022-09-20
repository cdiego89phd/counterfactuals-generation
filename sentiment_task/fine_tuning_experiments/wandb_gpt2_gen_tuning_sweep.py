import numpy as np
import pandas as pd
import torch
import argparse
import datetime
import yaml
import wandb
import sys
import openprompt
from sentiment_task import evaluation, generation, utils

from openprompt.prompts import ManualTemplate
from openprompt.plms.lm import LMTokenizerWrapper
import os


def generate_counterfactuals(yaml_file,
                             df_valset,
                             trained_lm,
                             tokenizer,
                             gen_params,
                             n_to_generate) -> pd.DataFrame:

    special_tokens = yaml_file['SPECIAL_TOKENS']
    map_labels = yaml_file['MAP_LABELS']
    generation_prompt = yaml_file['GENERATION_PROMPT']

    # wrap the datasets with the prompt template
    df_valset["wrapped_input"] = df_valset.apply(lambda row: utils.wrap_dataset_with_prompt(row,
                                                                                            generation_prompt,
                                                                                            map_labels,
                                                                                            special_tokens), axis=1)

    # prepare the data loader
    valset = generation.SentimentDataset(raw_dataframe=df_valset.copy(deep=True))
    valset.prepare_dataloader()
    print(f"{datetime.datetime.now()}: Valset prepared!")

    template_prompt = '{"placeholder":"text_a"}{"mask"}'
    prompt_template = ManualTemplate(text=template_prompt, tokenizer=tokenizer)
    tokenizer_wrapper = LMTokenizerWrapper
    val_data_loader = openprompt.PromptDataLoader(
        dataset=list(valset.get_dataset().values()),
        tokenizer=tokenizer,
        template=prompt_template,
        tokenizer_wrapper_class=tokenizer_wrapper
    )

    # set Random seed
    torch.manual_seed(yaml_file["SEED"])
    np.random.seed(yaml_file["SEED"])

    counter_generator = generation.CounterGenerator(prompt_template,
                                                    trained_lm,
                                                    val_data_loader,
                                                    valset,
                                                    gen_params)
    print(f"{datetime.datetime.now()}: Begin of generation...")

    # the generated counterfactuals are held inside the counter_generator object
    counter_generator.perform_generation(tokenizer, n_to_generate=n_to_generate)
    generated = counter_generator.dataframe_from_dataset(n_to_generate=n_to_generate)

    return generated


def run_agent(args, yaml_file):

    fold = yaml_file['FOLD']
    lm_name = yaml_file['LM_NAME']
    base_model = yaml_file['BASE_MODEL']
    task_name = yaml_file['TASK_NAME']
    prompt_id = yaml_file['PROMPT_ID']
    dataset_path = yaml_file['DATASET_PATH']
    special_tokens = yaml_file['SPECIAL_TOKENS']
    classifier_name = yaml_file['CLASSIFIER_NAME']
    n_to_generate = yaml_file['N_TO_GENERATE']
    # run_name = f"{lm_name}@prompt-{prompt_id}@fold-{fold}@{task_name}"

    tokenizer, _, _ = utils.load_gpt2_objects(base_model, special_tokens)
    model_local_path = f"{yaml_file['MODEL_DIR']}/{lm_name}"
    trained_lm = utils.load_gpt2_from_local(model_local_path)

    # load classifier for the evaluation
    classification_tools = utils.prepare_classifier(classifier_name)
    print(f"{datetime.datetime.now()}: Classifier prepared!")

    # load the dataset (we only use the valset)
    _, df_valset, _ = utils.load_dataset(f"{dataset_path}/fold_{fold}/")
    if args.debug_mode:
        df_valset = df_valset[:10]
    print(f"# of instances in the valset:{len(df_valset)}")

    with wandb.init(settings=wandb.Settings(console='off')):
        gen_params = wandb.config
        gen_params["do_sample"] = True
        print(f"Running generation with run:{wandb.run.name}")

        gen_valset = generate_counterfactuals(yaml_file,
                                              df_valset,
                                              trained_lm,
                                              tokenizer,
                                              gen_params,
                                              n_to_generate)
        print(f"{datetime.datetime.now()}: Generation completed!")

        evaluator = evaluation.SentimentEvaluator(classification_tools["tokenizer"],
                                                  classification_tools["classifier"],
                                                  classification_tools["label_map"])

        eval_valset, n_nan = evaluator.clean_evalset(gen_valset)
        evaluator.infer_predictions(eval_valset, n_generated=n_to_generate)
        lf_score = evaluator.calculate_lf_score(eval_valset)
        conf_score = evaluator.get_conf_score_pred()
        blue_mean, blue_var, _, _ = evaluator.calculate_bleu_score(eval_valset, n_to_generate)
        blue_corpus = evaluator.calculate_bleu_corpus(eval_valset, n_to_generate)

        wandb.log({"lf_score": lf_score,
                   "conf_score": conf_score,
                   "bleu_mean": blue_mean,
                   "bleu_var": blue_var,
                   "bleu_corpus": blue_corpus,
                   "n_nan": n_nan})
        print(f"{datetime.datetime.now()}: Evaluation completed!")

        return


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

    # e.g. SETTING_NAME = "tuning_gen_hypers_prompt-1_fold-0.yaml"
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
    # lm_name = parsed_yaml_file['LM_NAME']
    # prompt_id = parsed_yaml_file['PROMPT_ID']
    # special_tokens = parsed_yaml_file['SPECIAL_TOKENS']
    # classifier_name = parsed_yaml_file['CLASSIFIER_NAME']

    n_sweep_runs = parsed_yaml_file['N_SWEEP_RUNS']

    print(f"{datetime.datetime.now()}: Begin GEN TUNING for fold:{fold}")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)
    sweep_id = f"cdiego89/counterfactuals-generation/{args.sweep_id}"
    print(f"Sweep id:{sweep_id}")

    try:
        wandb.agent(sweep_id, function=lambda: run_agent(args, parsed_yaml_file), count=n_sweep_runs)

    except wandb.errors.CommError:
        print(f"wandb.errors.CommError: could not find sweep: {sweep_id}")
        sys.exit()

    print(f"{datetime.datetime.now()}: End GEN TUNING for fold:{fold}")
    sys.exit()


if __name__ == "__main__":
    os.environ['WANDB_CONSOLE'] = 'off'  # this will prevent the sweep to finish with no errors
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["CUDA_VISIBLE_DEVICES"] = 1
    main()
