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

    parser.add_argument(
        "--metrics",
        default=None,
        type=str,
        required=True,
        help="The metrics to compute. Include in the string g for goal-orientdness; b for BLEU; c for closeness; "
             "d for diversity."
    )

    args = parser.parse_args()

    # extract cfg from filename
    cfgs = args.results_filename.split('@')

    if len(cfgs) > 1:
        prompt_template = cfgs[1].split("-")[1]
        fold = cfgs[2].split("-")[1]
    else:
        prompt_template = "None"
        fold = "0"

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

    metrics_dict = {"n_nan": n_nan}

    # goal-orientdness metrics
    if "g" in args.metrics:
        evaluator.infer_predictions(eval_set, n_counter_generated)
        lf_score = evaluator.calculate_lf_score(eval_set)
        conf_score = evaluator.get_conf_score_pred()
        print(f"{datetime.datetime.now()}: LF score calculated!\n")

        # update dict
        metrics_dict["lf_score"] = lf_score
        metrics_dict["conf_score"] = conf_score

    # BLEU metrics
    if "b" in args.metrics:
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

        # update dict
        metrics_dict["bleu_corpus"] = bleu_corpus
        metrics_dict["bleu_corpus@1"] = bleu_corpus_1
        metrics_dict["bleu_corpus@2"] = bleu_corpus_2
        metrics_dict["bleu_corpus@3"] = bleu_corpus_3
        metrics_dict["bleu_corpus@4"] = bleu_corpus_4
        metrics_dict["bleu_mean"] = bleu_mean
        metrics_dict["bleu_mean@1"] = bleu_mean_1
        metrics_dict["bleu_mean@2"] = bleu_mean_2
        metrics_dict["bleu_mean@3"] = bleu_mean_3
        metrics_dict["bleu_mean@4"] = bleu_mean_4
        metrics_dict["bleu_spear"] = bleu_spear
        metrics_dict["bleu_pears"] = bleu_pears
        metrics_dict["bleu_var"] = bleu_var
        metrics_dict["bleu_var@1"] = bleu_var_1
        metrics_dict["bleu_var@2"] = bleu_var_2
        metrics_dict["bleu_var@3"] = bleu_var_3
        metrics_dict["bleu_var@4"] = bleu_var_4

    # closeness metrics
    if "c" in args.metrics:
        lev_dist_mean, lev_dist_var, lev_spear, lev_pears = evaluator.calculate_lev_dist(eval_set, n_counter_generated,
                                                                                         calculate_corr=calculate_corr)
        zss_dist_mean, zss_dist_var, zss_spear, zss_pears = evaluator.calculate_zss_dist(eval_set, n_counter_generated,
                                                                                         calculate_corr=calculate_corr)
        print(f"{datetime.datetime.now()}: Distances scores calculated!\n")

        # update dict
        metrics_dict["lev_dist_mean"] = lev_dist_mean
        metrics_dict["lev_dist_var"] = lev_dist_var
        metrics_dict["lev_spear"] = lev_spear
        metrics_dict["lev_pears"] = lev_pears
        metrics_dict["zss_dist_mean"] = zss_dist_mean
        metrics_dict["zss_dist_var"] = zss_dist_var
        metrics_dict["zss_spear"] = zss_spear
        metrics_dict["zss_pears"] = zss_pears

    if "d" in args.metrics:
        self_bleu_mean, self_bleu_var, self_bleu_spear, self_bleu_pears = evaluator.calculate_self_bleu(
            eval_set, n_counter_generated, weights=None, calculate_corr=True)  # it calculates the 4-grams
        print(f"{datetime.datetime.now()}: self-BLEU score calculated!\n")

        # update dict
        metrics_dict["self_bleu_mean"] = self_bleu_mean
        metrics_dict["self_bleu_var"] = self_bleu_var
        metrics_dict["self_bleu_spear"] = self_bleu_spear
        metrics_dict["self_bleu_pears"] = self_bleu_pears

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    # report all to wandb
    with wandb.init(settings=wandb.Settings(console='off'),
                    project=args.wandb_project):

        wandb.run.name = f"{args.eval_task_name}@{args.results_filename}"

        metrics_dict["avg_len_counter"] = np.mean(eval_set["counter_size"].values)
        metrics_dict["avg_len_generated_counter"] = np.mean(eval_set["generated_counter_size"].values)
        metrics_dict["lm_name"] = args.lm_name
        metrics_dict["eval_task_name"] = args.eval_task_name
        metrics_dict["prompt_template"] = prompt_template
        metrics_dict["fold"] = fold
        wandb.log(metrics_dict)

    print(f"{datetime.datetime.now()}: End of the evaluation with filename:{args.generation_path}")
    sys.exit()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
