import datetime
import argparse
import pandas as pd
import numpy as np
import sys
import os
from sentiment_task import evaluation
import utils


def evaluate(args, results_filename):
    # open the file with the results
    results_table = pd.read_csv(f"{args.generation_path}{results_filename}", sep='\t')

    # load classifier
    classification_tools = utils.prepare_sentiment_classifier(args.classifier_name)

    # prepare the evaluator
    evaluator = evaluation.SentimentEvaluator(classification_tools["tokenizer"],
                                              classification_tools["classifier"],
                                              classification_tools["label_map"])

    # run evaluation
    results_table.rename(columns={"label": "label_counter", "text": "generated_counter_0"}, inplace=True)
    eval_set, n_nan = evaluator.clean_evalset(results_table)  # remove the Nan counterfactuals
    print(f"Removed {n_nan} null counterfactuals!")

    # measures the label flip score
    evaluator.infer_predictions(eval_set, 1)
    lf_score = evaluator.calculate_lf_score(eval_set)
    conf_score = evaluator.get_conf_score_pred()
    print(f"{lf_score};{conf_score};")
    print(f"{datetime.datetime.now()}: LF score calculated!\n")

    avg_len_examples = np.mean([len(el.split(" ")) for el in eval_set["example"]])
    avg_len_counters = np.mean([len(el.split(" ")) for el in eval_set["generated_counter_0"]])

    print(f"Avg len seed review:{avg_len_examples}")
    print(f"Avg len counter review:{avg_len_counters}")


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

    args = parser.parse_args()

    files = args.results_filename.split(";")
    if args.results_filename == "":
        # read all files from folder
        files = [f for f in os.listdir(args.generation_path)
                 if os.path.isfile(os.path.join(args.generation_path, f))]

    for gen_file in files:
        print(f"{datetime.datetime.now()}: Beginning of evaluation with filename:{gen_file}")
        evaluate(args, gen_file)
        print(f"{datetime.datetime.now()}: End of evaluation with filename:{gen_file}")
        print("########################################################################")
        print("")

    sys.exit()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
