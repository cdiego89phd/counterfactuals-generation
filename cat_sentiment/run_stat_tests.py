import pandas as pd
import argparse
import transformers
import datetime
from statsmodels.stats.contingency_tables import mcnemar


def build_contingency(pred_1: list, pred_2: list, gold_labels: list) -> list:

    check_correct = {True: "C",
                     False: "I"}

    contingency = {"CC": 0,
                   "CI": 0,
                   "IC": 0,
                   "II": 0}

    for p1, p2, gold in zip(pred_1, pred_2, gold_labels):
        key = check_correct[p1 == gold] + check_correct[p2 == gold]
        contingency[key] += 1

    return [[contingency["CC"], contingency["CI"]], [contingency["IC"], contingency["II"]]]


def run_classification(classifier_path: str,
                       classifier_name: str,
                       base_classifier_name: str,
                       testset: pd.DataFrame) -> list:

    # mapping of the labels
    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}

    # load classifier and tokenizer
    classifier_path = f"{classifier_path}/{classifier_name}"
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_path,
                                                                                 id2label=id2label,
                                                                                 label2id=label2id,
                                                                                 num_labels=2,
                                                                                 local_files_only=True,
                                                                                 )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_classifier_name)
    tokenizer_kwargs = {'padding': True,
                        'truncation': True,
                        'max_length': tokenizer.model_max_length}
    pipeline_classifier = transformers.pipeline("text-classification",
                                                model=classifier,
                                                tokenizer=tokenizer,
                                                device=0,
                                                **tokenizer_kwargs)
    print(f"{datetime.datetime.now()}: Classifier pipeline built!")

    raw_preds = pipeline_classifier(list(testset["text"].values))
    pred_labels = [label2id[el['label']] for el in raw_preds]

    return pred_labels


def run_mcnemar(args: argparse.Namespace) -> None:

    # load test set
    test_data = pd.read_csv(f"{args.evaluation_path}{args.test_filename}", sep='\t')
    if args.debug_mode:
        test_data = test_data[:10]

    pred_1 = run_classification(args.classifier_path, args.classifier_name_1, args.base_classifier_name, test_data)
    pred_2 = run_classification(args.classifier_path, args.classifier_name_2, args.base_classifier_name, test_data)
    gold_labels = test_data["labels"].values

    alpha_values = [0.05, 0.01]
    cont_table = build_contingency(pred_1, pred_2, gold_labels)

    test = mcnemar(cont_table, exact=True)
    print("Results with exact=True parameter:")
    for alpha in alpha_values:
        print(f"Significance with McNemar test at alpha:{alpha}: {test.pvalue < alpha} --> p-value:{test.pvalue}")

    print()
    test = mcnemar(cont_table, exact=False, correction=True)
    print("Results with exact=True parameter:")
    for alpha in alpha_values:
        print(f"Significance with McNemar test at alpha:{alpha}: {test.pvalue < alpha} --> p-value:{test.pvalue}")


def main():

    # read params from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_path",
        default=None,
        type=str,
        required=True,
        help="The path where the test set is stored."
    )

    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        required=True,
        help="The filename of the results."
    )

    parser.add_argument(
        "--base_classifier_name",
        default=None,
        type=str,
        required=True,
        help="The name of the base classifier."
    )

    parser.add_argument(
        "--classifier_path",
        default=None,
        type=str,
        required=True,
        help="The path where the classifier is stored."
    )

    parser.add_argument(
        "--classifier_name_1",
        default=None,
        type=str,
        required=True,
        help="The classifier_1 to compare."
    )

    parser.add_argument(
        "--classifier_name_2",
        default=None,
        type=str,
        required=True,
        help="The classifier_2 to compare."
    )

    parser.add_argument(
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )

    args = parser.parse_args()

    run_mcnemar(args)


if __name__ == "__main__":
    main()
