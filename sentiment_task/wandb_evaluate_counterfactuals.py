import datetime
import argparse
import pandas as pd
import wandb
import sys
import os
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
        "--cuda_device",
        default=None,
        type=str,
        required=True,
        help="The id of the cuda device."
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
    cfgs = args.results_filename.split('_')
    prompt_template = cfgs[1]
    fold = cfgs[2][-1]

    print(f"{datetime.datetime.now()}: Beginning of evaluation with filename:{args.generation_path}")

    # open the file with the results
    results_table = pd.read_csv(f"{args.generation_path}{args.results_filename}.csv", sep='\t')

    # load classifier
    classification_tools = utils.prepare_classifier(args.classifier_name)

    # prepare the evaluator
    evaluator = evaluation.SentimentEvaluator(classification_tools["tokenizer"],
                                              classification_tools["classifier"],
                                              classification_tools["label_map"],
                                              int(args.cuda_device))

    # run evaluation
    eval_set, n_nan = evaluator.clean_evalset(results_table)  # remove the Nan counterfactuals
    evaluator.infer_predictions(eval_set)
    lf_score = evaluator.calculate_lf_score(eval_set)
    conf_score = evaluator.get_conf_score_pred()
    blue_corpus = evaluator.calculate_blue_corpus(eval_set)
    blue_mean, blue_var = evaluator.calculate_blue_score(eval_set)

    # initialize WANDB logging system
    wandb.login(relogin=True, key=args.wandb_key)

    # report all to wandb
    with wandb.init(settings=wandb.Settings(console='off'),
                    project=args.wandb_project):

        wandb.run.name = f"{args.eval_task_name}@{args.results_filename}"
        d = {"n_nan": n_nan,
             "lf_score": lf_score,
             "conf_score": conf_score,
             "blue_corpus": blue_corpus,
             "blue_mean": blue_mean,
             "blue_var": blue_var,
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
