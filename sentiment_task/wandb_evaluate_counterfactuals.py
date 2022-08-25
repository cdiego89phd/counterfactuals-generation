import datetime
import argparse
import pandas as pd
import wandb
import sys
import os
import numpy as np
from sentiment_task import evaluation, utils


def main():

    # read params from command line
    parser = argparse.ArgumentParser()
    # SETTINGS_PATH = "/home/diego/counterfactuals-generation/sentiment_task/fine_tuning_experiments/generation/"
    parser.add_argument(
        "--generation_path",
        default=None,
        type=str,
        required=True,
        help="The path where results (generated counterfactuals) are stored."
    )

    parser.add_argument(
        "--results_filename",
        default=None,
        type=str,
        required=True,
        help="The filename of the results."
    )

    parser.add_argument(
        "--classifier_name",
        default=None,
        type=str,
        required=True,
        help="The name of the sentiment classifier."
    )

    parser.add_argument(
        "--calculate_corr",
        default=False,
        type=bool,
        required=False,
        help="Whether to calculate correlation for all the reported metrics."
    )

    parser.add_argument(
        "--n_counter_generated",
        default=False,
        type=int,
        required=False,
        help="The number of counterfactuals generated in the output file to process."
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
        help="The name of the wandb project."
    )

    parser.add_argument(
        "--lm_name",
        default=None,
        type=str,
        required=True,
        help="The name of the language model used to generate the counterfactuals"
    )

    parser.add_argument(
        "--eval_task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the evaluation task (e.g. fine_tuning_experiments)."
    )

    args = parser.parse_args()

    # extract cfg from filename
    cfgs = args.results_filename.split('@')
    prompt_template = cfgs[1].split("-")[1]
    fold = cfgs[2].split("-")[1]

    print(f"{datetime.datetime.now()}: Beginning of evaluation with filename:{args.generation_path}")

    # open the file with the results
    results_table = pd.read_csv(f"{args.generation_path}{args.results_filename}", sep='\t')

    # load classifier
    classification_tools = utils.prepare_classifier(args.classifier_name)

    # Whether to calculate correlation for all the reported metrics
    calculate_corr = args.calculate_corr

    # The number of counterfactuals generated in the output file to process.
    n_counter_generated = args.n_counter_generated

    # prepare the evaluator
    evaluator = evaluation.SentimentEvaluator(classification_tools["tokenizer"],
                                              classification_tools["classifier"],
                                              classification_tools["label_map"])

    # run evaluation
    eval_set, n_nan = evaluator.clean_evalset(results_table)  # remove the Nan counterfactuals
    eval_set = evaluator.retrieve_sizes(eval_set, n_counter_generated)

    evaluator.infer_predictions(eval_set, n_counter_generated)
    lf_score = evaluator.calculate_lf_score(eval_set)
    conf_score = evaluator.get_conf_score_pred()
    print(f"{datetime.datetime.now()}: LF score calculated!\n")

    bleu_corpus = evaluator.calculate_bleu_corpus(eval_set, n_counter_generated)
    bleu_corpus_1 = evaluator.calculate_bleu_corpus(eval_set, n_counter_generated, weights=(1, 0, 0, 0))
    bleu_corpus_2 = evaluator.calculate_bleu_corpus(eval_set, n_counter_generated, weights=(0, 1, 0, 0))
    bleu_corpus_3 = evaluator.calculate_bleu_corpus(eval_set, n_counter_generated, weights=(0, 0, 1, 0))
    bleu_corpus_4 = evaluator.calculate_bleu_corpus(eval_set, n_counter_generated, weights=(0, 0, 0, 1))
    print(f"{datetime.datetime.now()}: BLEU corpus score calculated!\n")

    bleu_mean, bleu_var, bleu_spear, bleu_pears = evaluator.calculate_bleu_score(
        eval_set, n_counter_generated, calculate_corr=calculate_corr)
    bleu_mean_1, bleu_var_1, _, _ = evaluator.calculate_bleu_score(eval_set, n_counter_generated, weights=(1, 0, 0, 0))
    bleu_mean_2, bleu_var_2, _, _ = evaluator.calculate_bleu_score(eval_set, n_counter_generated, weights=(0, 1, 0, 0))
    bleu_mean_3, bleu_var_3, _, _ = evaluator.calculate_bleu_score(eval_set, n_counter_generated, weights=(0, 0, 1, 0))
    bleu_mean_4, bleu_var_4, _, _ = evaluator.calculate_bleu_score(eval_set, n_counter_generated, weights=(0, 0, 0, 1))
    print(f"{datetime.datetime.now()}: BLEU score calculated!\n")

    self_bleu_mean, self_bleu_var, self_bleu_spear, self_bleu_pears = evaluator.calculate_self_bleu(
        eval_set, n_counter_generated, weights=None, calculate_corr=True)  # it calculates the 4-grams
    print(f"{datetime.datetime.now()}: self-BLEU score calculated!\n")

    lev_dist_mean, lev_dist_var, lev_spear, lev_pears = evaluator.calculate_lev_dist(eval_set, n_counter_generated,
                                                                                     calculate_corr=calculate_corr)
    zss_dist_mean, zss_dist_var, zss_spear, zss_pears = evaluator.calculate_zss_dist(eval_set, n_counter_generated,
                                                                                     calculate_corr=calculate_corr)
    print(f"{datetime.datetime.now()}: Distances score calculated!\n")

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    # report all to wandb
    with wandb.init(settings=wandb.Settings(console='off'),
                    project=args.wandb_project):

        wandb.run.name = f"{args.eval_task_name}@{args.results_filename}"
        d = {"n_nan": n_nan,
             "lf_score": lf_score,
             "conf_score": conf_score,
             "bleu_corpus": bleu_corpus,
             "bleu_corpus@1": bleu_corpus_1,
             "bleu_corpus@2": bleu_corpus_2,
             "bleu_corpus@3": bleu_corpus_3,
             "bleu_corpus@4": bleu_corpus_4,
             "bleu_mean": bleu_mean,
             "bleu_mean@1": bleu_mean_1,
             "bleu_mean@2": bleu_mean_2,
             "bleu_mean@3": bleu_mean_3,
             "bleu_mean@4": bleu_mean_4,
             "bleu_spear": bleu_spear,
             "bleu_pears": bleu_pears,
             "bleu_var": bleu_var,
             "bleu_var@1": bleu_var_1,
             "bleu_var@2": bleu_var_2,
             "bleu_var@3": bleu_var_3,
             "bleu_var@4": bleu_var_4,
             "self_bleu_mean": self_bleu_mean,
             "self_bleu_var": self_bleu_var,
             "self_bleu_spear": self_bleu_spear,
             "self_bleu_pears": self_bleu_pears,
             "lev_dist_mean": lev_dist_mean,
             "lev_dist_var": lev_dist_var,
             "lev_spear": lev_spear,
             "lev_pears": lev_pears,
             "zss_dist_mean": zss_dist_mean,
             "zss_dist_var": zss_dist_var,
             "zss_spear": zss_spear,
             "zss_pears": zss_pears,
             "avg_len_counter": np.mean(eval_set["counter_size"].values),
             "avg_len_generated_counter": np.mean(eval_set["generated_counter_size"].values),
             "lm_name": args.lm_name,
             "eval_task_name;": args.eval_task_name,
             "prompt_template": prompt_template,
             "fold": fold}
        wandb.log(d)

    print(f"{datetime.datetime.now()}: End of the evaluation with filename:{args.generation_path}")
    sys.exit()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
