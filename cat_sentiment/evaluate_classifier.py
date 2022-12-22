from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime
import argparse
import pandas as pd
import numpy as np
import transformers
import wandb
import sys
import os


def compute_metrics(true_labels, pred_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    acc = accuracy_score(true_labels, pred_labels)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def evaluate(args):
    # open the file with the results
    test_data = pd.read_csv(f"{args.evaluation_path}{args.test_filename}", sep='\t')
    if args.debug_mode:
        test_data = test_data[:10]

    # mapping of the labels
    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}

    # load classifier and tokenizer
    classifier_path = f"{args.classifier_path}/{args.classifier_name}"
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_path,
                                                                                 id2label=id2label,
                                                                                 label2id=label2id,
                                                                                 num_labels=2,
                                                                                 local_files_only=True,
                                                                                 )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_classifier_name)
    tokenizer_kwargs = {'padding': True,
                        'truncation': True,
                        'max_length': tokenizer.model_max_length}
    pipeline_classifier = transformers.pipeline("text-classification",
                                                model=classifier,
                                                tokenizer=tokenizer,
                                                device=0,
                                                **tokenizer_kwargs)
    print(f"{datetime.datetime.now()}: Classifier pipeline built!")

    raw_preds = pipeline_classifier(list(test_data["text"].values))
    pred_labels = [label2id[el['label']] for el in raw_preds]
    pred_conf = [el['score'] for el in raw_preds]
    metrics_dict = compute_metrics(list(test_data["labels"].values), pred_labels)

    metrics_dict["review_len"] = np.mean(test_data['review_len'].values)
    metrics_dict["pred_conf"] = np.mean(pred_conf)

    print("Now logging into wandb...")
    # report all to wandb
    with wandb.init(settings=wandb.Settings(console='off'),
                    project=args.wandb_project):
        wandb.run.name = f"{args.classifier_name}"
        wandb.log(metrics_dict)
        wandb.finish
    sys.exit()


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
        "--classifier_path",
        default=None,
        type=str,
        required=True,
        help="The path where the classifier is stored."
    )

    parser.add_argument(
        "--classifier_name",
        default=None,
        type=str,
        required=True,
        help="The name of the sentiment classifier."
    )

    parser.add_argument(
        "--base_classifier_name",
        default=None,
        type=str,
        required=True,
        help="The name of the base classifier."
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
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )

    args = parser.parse_args()

    print(f"{datetime.datetime.now()}: Beginning of evaluation with model:{args.classifier_name}")
    evaluate(args)
    print(f"{datetime.datetime.now()}: End of evaluation with filename:{args.classifier_name}")
    print("########################################################################")
    print("")


if __name__ == "__main__":
    main()
