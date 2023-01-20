import utils
import argparse
import datetime
import yaml
import wandb
import sys
from nli_task import generator
import evaluation

import os


def run_agent(args, yaml_file):

    fold = yaml_file['FOLD']
    lm_name = yaml_file['LM_NAME']
    base_model = yaml_file['BASE_MODEL']
    dataset_path = yaml_file['DATASET_PATH']
    special_tokens = yaml_file['SPECIAL_TOKENS']
    classifier_name = yaml_file['CLASSIFIER_NAME']
    n_to_generate = yaml_file['N_TO_GENERATE']

    tokenizer = utils.load_tokenizer(base_model, special_tokens)
    model_local_path = f"{yaml_file['MODEL_DIR']}/{lm_name}"
    trained_lm, _ = utils.load_causal_model_from_local(model_local_path)

    # load classifier for the evaluation
    classification_tools = utils.prepare_nli_classifier(classifier_name)
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

        gen_valset = generator.generate_counterfactuals(yaml_file,
                                                        df_valset,
                                                        trained_lm,
                                                        tokenizer,
                                                        gen_params,
                                                        n_to_generate)
        print(f"{datetime.datetime.now()}: Generation completed!")

        evaluator = evaluation.NLIEvaluator(classification_tools["tokenizer"],
                                            classification_tools["classifier"],
                                            classification_tools["label_map"],
                                            gen_valset)
        n_nan = evaluator.clean_evalset()
        evaluator.prepare_batches(args.n_batches, n_to_generate)

        evaluator.infer_predictions(n_to_generate)
        lf_score = evaluator.calculate_lf_score()
        blue_mean, blue_var, _, _ = evaluator.calculate_bleu_score(n_to_generate)
        blue_corpus = evaluator.calculate_bleu_corpus(n_to_generate)

        wandb.log({"lf_score": lf_score,
                   "bleu_mean": blue_mean,
                   "bleu_var": blue_var,
                   "bleu_corpus": blue_corpus,
                   "n_nan": n_nan})
        print(f"{datetime.datetime.now()}: Evaluation completed!")

        return


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

    # e.g. SETTING_NAME = "tuning_gen_hypers_prompt-1_fold-0.yaml"
    parser.add_argument(
        "--setting_name",
        default=None,
        type=str,
        required=True,
        help="The name of yaml file where to load the setting from."
    )

    parser.add_argument(
        "--n_batches",
        default=1,
        type=int,
        required=True,
        help="The number of batches to use."
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

    print(f"{datetime.datetime.now()}: Begin GEN TUNING for fold:{fold}")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)
    sweep_id = f"{args.wandb_project}/{args.sweep_id}"
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
