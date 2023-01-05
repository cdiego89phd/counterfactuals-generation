import torch
import pandas as pd
import datasets
import argparse
import transformers
import datetime
import tabulate
from fairseq.data.data_utils import collate_tokens


MODELS = ["roberta.large.mnli",
          "cross-encoder/nli-distilroberta-base",
          "facebook/bart-large-mnli",
          "microsoft/deberta-v3-base",
          "microsoft/deberta-v3-large",
          "microsoft/deberta-large-mnli",
          "cross-encoder/nli-deberta-v3-base",
          "cross-encoder/nli-deberta-v3-large",
          "textattack/bert-base-uncased-snli",
          "textattack/albert-base-v2-snli",
          "textattack/distilbert-base-cased-snli",
          "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
          "pepa/bigbird-roberta-large-snli"
          ]


def extract_prems(row):
    if row["task"] == "RP":
        return row["counter_prem"]
    else:
        return row["original_prem"]


def extract_hyps(row):
    if row["task"] == "RH":
        return row["counter_hyp"]
    else:
        return row["original_hyp"]


def generate_batches(data, n):
    batch_size = len(data)//n
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
    return data


def evaluate_classifier(preds, labels, eval_m):
    # evaluates a classifier
    metrics = {"precision": eval_m["precision"].compute(predictions=preds, references=labels,
                                                        average="micro")["precision"],
               "recall": eval_m["recall"].compute(predictions=preds, references=labels, average="micro")["recall"],
               "f1": eval_m["f1"].compute(predictions=preds, references=labels, average="micro")["f1"],
               "accuracy": eval_m["accuracy"].compute(predictions=preds, references=labels)["accuracy"],
               }
    return metrics


def run_classifier(model_name: str,
                   eval_data: pd.DataFrame,
                   eval_batch: list,
                   eval_metrics: dict,
                   n_batches: int
                   ) -> dict:

    if model_name == "roberta.large.mnli":
        class_map = {"contradiction": 0,
                     "neutral": 1,
                     "entailment": 2
                     }
        gold_labels = [class_map[el] for el in eval_data["counter_label"]]

        model = torch.hub.load('pytorch/fairseq', model_name)
        model.cuda()
        model.eval()
        data = collate_tokens(
            [model.encode(pair[0], pair[1]) for pair in eval_batch], pad_idx=1
        )
        batches = generate_batches(data, n_batches)
        predictions = []
        for batch in batches:
            predictions += model.predict('mnli', batch).argmax(dim=1)
    else:
        class_map = {"contradiction": 0,
                     "entailment": 1,
                     "neutral": 2
                     }
        gold_labels = [class_map[el] for el in eval_data["counter_label"]]

        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        batches = generate_batches(eval_batch, n_batches)

        model.cuda()
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in batches:
                tokenized_batch = tokenizer(batch,  padding=True, truncation=True, return_tensors="pt")
                scores = model(**tokenized_batch.to('cuda')).logits
                predictions += [score_max for score_max in scores.argmax(dim=1)]

    print(f"Evaluating {len(predictions)} predictions...")
    model_result = evaluate_classifier(predictions, gold_labels, eval_metrics)
    print(f"{datetime.datetime.now()} RESULTS FOR MODEL:{model_name}")
    print(model_result)

    del model
    torch.cuda.empty_cache()

    model_result["model_name"] = model_name
    return model_result


def main():

    # read params from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to_debug",
        default=0,
        type=int,
        required=True,
        help="Whether to run script in debug mode."
    )

    parser.add_argument(
        "--n_to_debug",
        default=12,
        type=int,
        required=False,
        help="The number of dataset instance to classify in debug mode."
    )

    parser.add_argument(
        "--n_batches",
        default=1,
        type=int,
        required=True,
        help="The number of batches to use."
    )

    args = parser.parse_args()

    eval_metrics = {"precision": datasets.load_metric("precision"),
                    "recall": datasets.load_metric("recall"),
                    "f1": datasets.load_metric("f1"),
                    "accuracy": datasets.load_metric("accuracy")
                    }

    trainset = pd.read_csv("cad_flickr_nli/fold_0/training_set.tsv", sep='\t')
    valset = pd.read_csv("cad_flickr_nli/fold_0/val_set.tsv", sep='\t')
    testset = pd.read_csv("cad_flickr_nli/fold_0/test_set.tsv", sep='\t')
    eval_data = pd.concat([trainset, valset, testset], ignore_index=True)

    if args.to_debug:
        eval_data = eval_data[:args.n_to_debug]
    eval_data.reset_index(inplace=True, drop=True)

    eval_data["premise"] = eval_data.apply(lambda row: extract_prems(row), axis=1)
    eval_data["hypothesis"] = eval_data.apply(lambda row: extract_hyps(row), axis=1)

    eval_batch = [[p, h] for p, h in zip(eval_data["premise"].values, eval_data["hypothesis"].values)]

    print(f"# of instances to classify:{len(eval_data)}")

    dict_comparison = {"model_name": [],
                       "accuracy": [],
                       "precision": [],
                       "recall": [],
                       "f1": []
                       }
    for model_name in MODELS:
        print(f"{datetime.datetime.now()}: Begin evaluation for model:{model_name}")
        results = run_classifier(model_name, eval_data, eval_batch, eval_metrics, args.n_batches)
        for key in results:
            dict_comparison[key].append(results[key])
        print("##############################################################")
        print()
        df_comparison = pd.DataFrame(data=dict_comparison)
        print(tabulate.tabulate(df_comparison, headers='keys', tablefmt='psql', showindex=False))


if __name__ == "__main__":
    main()

