import argparse
import pandas as pd
import transformers
from sentiment_task import utils


def classify_text(text_to_classify, model, label_map):
    if len(text_to_classify) > 512:
        return label_map[model(text_to_classify[:511])[0]['label']]
    else:
        return label_map[model(text_to_classify)[0]['label']]


def evaluate(args):
    # open the file with the results
    counter_data = pd.read_csv(f"{args.generation_path}{args.results_filename}", sep='\t')
    n_nan = counter_data['generated_counter_0'].isna().sum()
    print(f"# of nan values removed in generated counterfactuals:{n_nan}")
    counter_data.dropna(inplace=True)

    if args.debug_mode:
        counter_data = counter_data[:3]
        # TODO remove this
        counter_data["generated_counter_1"] = counter_data["generated_counter_0"].values
        counter_data["generated_counter_2"] = counter_data["generated_counter_0"].values

    # load classifier
    classification_tools = utils.prepare_classifier(args.classifier_name)

    # prepare the classifier
    classifier = transformers.pipeline(
        task="sentiment-analysis",
        model=classification_tools['classifier'],
        tokenizer=classification_tools['tokenizer'],
        framework="pt",
        device=0)
    label_map = classification_tools["label_map"]

    texts = []
    labels = []

    for _, row in counter_data.iterrows():
        target_label = row['label_counter']
        filtered_text = row['generated_counter_0']
        pred_label = int(classify_text(filtered_text, classifier, label_map))

        if pred_label != target_label:
            # look for the following generated counterfactuals
            for i in range(1, args.n_generated):
                pred = int(classify_text(row[f'generated_counter_{i}'], classifier, label_map))
                if pred == target_label:
                    filtered_text = row[f'generated_counter_{i}']
                    pred_label = pred
                    break

        texts.append(filtered_text)
        labels.append(pred_label)

    out_label = args.results_filename.split("@")[1].split(".")[0]
    # print filtered counterfactuals
    d = {"paired_id": counter_data["paired_id"].values,
         "example": counter_data["example"].values,
         "label_ex": counter_data["label_ex"].values,
         "counterfactual": [None for i in range(len(counter_data))],
         "generated_counter_0": texts,
         "label_counter": labels}

    # print filtered counterfactuals
    pd.DataFrame(data=d).to_csv(f"{args.generation_path}filtered_counterfactual@{out_label}.csv",
                                sep='\t', header=True, index=False)

    # print filtered cat data
    d = {"text": texts, "labels": labels}
    filtered = pd.DataFrame(data=d)
    n_data = pd.read_csv(f"{args.generation_path}n_data.csv", sep='\t')
    n_data.drop(columns=["sentiment", "review_len"], inplace=True)
    training_data = pd.concat([n_data, filtered])
    n_nan = training_data['text'].isna().sum()
    print(f"# of nan values removed in trainset:{n_nan}")
    training_data.to_csv(f"{args.generation_path}filtered_cat_data_{out_label}.csv",
                         sep='\t', header=True, index=False)


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
        "--n_generated",
        default=None,
        type=int,
        required=True,
        help="The # of generated counterfactuals."
    )

    parser.add_argument(
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )

    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
