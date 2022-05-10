import torch
import transformers
import numpy as np
import pandas as pd
import yaml
import itertools
import sklearn
import nltk
import argparse
import datetime


class Evaluator:

    def __init__(self, s_dataset, s_tokenizer, s_model, s_label_dict, s_device):
        """Constructor of the evaluator
        @param: s_dataset The dataset for the sentiment classifier
        @param: s_tokenizer The tokenizer for the sentiment classifier
        @param: s_model The sentiment classifier
        @param: s_label_dict The mapping between classifier's outputs and labels
        """
        self.s_dataset = s_dataset
        self.s_tokenizer = s_tokenizer
        self.s_model = s_model
        self.s_label_dict = s_label_dict
        if torch.cuda.is_available():
            s_device = -1
        self.classifier = transformers.pipeline(
            task="sentiment-analysis",
            model=s_model,
            tokenizer=s_tokenizer,
            framework="pt",
            device=s_device)
        self.predicted_labels = []
        self.score_labels = []

        # remove some nan values in the generated counterfactuals
        print(f"# of nan values removed in generated counterfactuals:{self.s_dataset['generated_counter'].isna().sum()}")
        self.s_dataset = self.s_dataset.dropna()

    def infer_predictions(self):

        texts = self.s_dataset["generated_counter"].values

        for text_to_classify in texts:
            if len(text_to_classify) > 512:
                result = self.classifier(text_to_classify[:511])[0]
            else:
                result = self.classifier(text_to_classify)[0]

            self.predicted_labels.append(self.s_label_dict[result['label']])
            self.score_labels.append(result['score'])

    def lf_score(self):
        """Calculate the Label Flip Score (LFS)
        """
        y_desired = self.s_dataset["label_counter"].values
        return sklearn.metrics.accuracy_score(y_desired, self.predicted_labels)

    def get_conf_score_pred(self):
        return np.mean(self.score_labels)

    def blue_score(self):
        """Calculate the BLUE score for a pair of example-counter.

           Returns mean and variance of the BLUE scores.
        """
        BLEUscore = []
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
        true_counters = self.s_dataset["counterfactual"].values
        gen_counters = self.s_dataset["generated_counter"].values
        for true_counter, gen_counter in zip(true_counters, gen_counters):
            # example and counterfactual need to be tokenized first

            # the reference is the true counterfactual
            reference = nltk.tokenize.word_tokenize(true_counter)

            # the hypothesis is the generated counterfactual
            hypothesis = nltk.tokenize.word_tokenize(gen_counter)

            BLEUscore.append(nltk.translate.bleu_score.sentence_bleu([reference],
                                                                     hypothesis,
                                                                     smoothing_function=smoothing_function.method1))
        return np.mean(BLEUscore), np.var(BLEUscore)


def load_results(folds, results_path, results_name, params):
    results = {}
    for fold in folds:
        results[fold] = {}
        for pars in params:
            results[fold][pars] = pd.read_csv(f"{results_path}fold_{fold}/{results_name}{pars}.gen", sep='\t')
    return results


def evaluate_fold_results(f_res, o_file, fold, token, model, label_map, cuda_device):
    for cfg_gen in f_res:
        print(f"Cfg:{cfg_gen}")
        cfg_res = f_res[cfg_gen]
        evaluator = Evaluator(cfg_res, token, model, label_map, cuda_device)
        blue_score, var_blue_score = evaluator.blue_score()
        evaluator.infer_predictions()
        conf_score = evaluator.get_conf_score_pred()
        lf_score = evaluator.lf_score()

        o_str = f"{fold}\t{cfg_gen}\t{blue_score}\t{var_blue_score}\t{lf_score}\t{conf_score}\n"
        o_file.write(o_str)


def main():

    # load required args from command line
    parser = argparse.ArgumentParser()

    # SETTINGS_PATH = "/home/diego/counterfactuals-generation/sentiment_task/zs_gpt2_experiments/settings/"
    parser.add_argument(
        "--setting_path",
        default=None,
        type=str,
        required=True,
        help="The absolute path of the file settings."
    )

    # SETTING_NAME = "zs_prompt_1_validation.yaml"
    parser.add_argument(
        "--setting_name",
        default=None,
        type=str,
        required=True,
        help="The name of yaml file where to load the setting from."
    )

    # RESULTS_NAME = "zs_prompt_1_validation-"
    parser.add_argument(
        "--results_name",
        default=None,
        type=str,
        required=True,
        help="The name of the generated counterfactuals file."
    )

    parser.add_argument(
        "--evaluation_name",
        default=None,
        type=str,
        required=True,
        help="The name of the file where to output the evaluation."
    )

    parser.add_argument(
        "--prompt_id",
        default=None,
        type=str,
        required=True,
        help="The id of the prompt used for the generation."
    )

    parser.add_argument(
        "--classifier_name",
        default=None,
        type=str,
        required=True,
        help="Name of the sentiment classifier"
    )

    parser.add_argument(
        "--cuda_device",
        default=None,
        type=str,
        required=True,
        help="Specify the gpu device"
    )

    args = parser.parse_args()
    setting_yaml_file = open(f"{args.setting_path}{args.setting_name}")
    parsed_yaml_file = yaml.load(setting_yaml_file, Loader=yaml.FullLoader)

    # the following params will be included in a yaml file
    RESULTS_PATH = parsed_yaml_file['RESULTS_PATH']
    FOLDS = parsed_yaml_file['FOLDS']
    GEN_ARGS = parsed_yaml_file['GEN_ARGS']
    print("Evaluation's params read from yaml file")

    # load the classifier
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.classifier_name)
    lm = transformers.AutoModelForSequenceClassification.from_pretrained(args.classifier_name)
    classifier_label_map = {'NEGATIVE':  0, 'POSITIVE': 1}

    all_pars = sorted(GEN_ARGS)
    gen_grid = list(itertools.product(*(GEN_ARGS[par] for par in all_pars)))

    results_dict = load_results(FOLDS, RESULTS_PATH, args.results_name, gen_grid)
    print("Results loaded...")

    print(f"{datetime.datetime.now()}: Calculating metrics on results...")
    with open(RESULTS_PATH + args.evaluation_name, 'w') as outfile:
        # print file header
        outfile.write("fold\tcfg\tblue\tvar_blue\tlfs\tconf_score_pred\n")
        for foldd in results_dict:
            fold_results = results_dict[foldd]
            # add fold and cfg
            out_str = f"{foldd}\t"
            evaluate_fold_results(fold_results, outfile, foldd, tokenizer, lm, classifier_label_map, args.cuda_device)
            print(f"{datetime.datetime.now()}: fold {foldd} DONE.")

    # retrieve the best configuration for each fold
    df_results = pd.read_csv(f"{RESULTS_PATH}{args.evaluation_name}", sep='\t')
    print("Detecting best configuration for each fold")
    max_values_lfs = df_results.loc[df_results.reset_index().groupby(['fold'])['lfs'].idxmax()]
    max_values_blue = df_results.loc[df_results.reset_index().groupby(['fold'])['blue'].idxmax()]
    max_values_lfs.to_csv(f"{RESULTS_PATH}prompt_{args.prompt_id}_val_best_lfs.csv", sep="\t", header=True, index=False)
    max_values_blue.to_csv(f"{RESULTS_PATH}prompt_{args.prompt_id}_val_best_blue.csv", sep="\t", header=True, index=False)
    print("DONE")


if __name__ == "__main__":
    main()
