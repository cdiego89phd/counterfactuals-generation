import torch
import transformers
import numpy as np
import pandas as pd
import sklearn
import nltk


class SentimentEvaluator:

    def __init__(self, s_tokenizer, s_model, s_label_dict, s_device):
        """Constructor of the evaluator
        @param: s_dataset The dataset for the sentiment classifier
        @param: s_tokenizer The tokenizer for the sentiment classifier
        @param: s_model The sentiment classifier
        @param: s_label_dict The mapping between classifier's outputs and labels
        @param: device where to run the classification pipeline
        """
        self.s_tokenizer = s_tokenizer
        self.s_model = s_model
        self.s_label_dict = s_label_dict
        self.classifier = transformers.pipeline(
            task="sentiment-analysis",
            model=s_model,
            tokenizer=s_tokenizer,
            framework="pt",
            device=s_device)
        self.predicted_labels = []
        self.score_labels = []

    @staticmethod
    def clean_evalset(eval_dataset):
        """Remove some nan values in the generated counterfactuals"""

        print(f"# of nan values removed in generated counterfactuals:{eval_dataset['generated_counter'].isna().sum()}")
        return eval_dataset.dropna()

    def infer_predictions(self, eval_dataset):
        """Infer the labels for the counterfactuals"""

        texts = eval_dataset["generated_counter"].values

        for text_to_classify in texts:
            if len(text_to_classify) > 512:
                result = self.classifier(text_to_classify[:511])[0]
            else:
                result = self.classifier(text_to_classify)[0]

            self.predicted_labels.append(self.s_label_dict[result['label']])
            self.score_labels.append(result['score'])

    def calculate_lf_score(self, eval_dataset):
        """Calculate the Label Flip Score (LFS)"""

        y_desired = eval_dataset["label_counter"].values
        return sklearn.metrics.accuracy_score(y_desired, self.predicted_labels)

    def get_conf_score_pred(self):
        return np.mean(self.score_labels)

    @staticmethod
    def calculate_blue_score(eval_dataset):
        """Calculate the BLUE score for a pair of example-counter.

           Returns mean and variance of the BLUE scores.
        """
        BLEUscore = []
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
        true_counters = eval_dataset["counterfactual"].values
        gen_counters = eval_dataset["generated_counter"].values
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
        evaluator = SentimentEvaluator(cfg_res, token, model, label_map, cuda_device)
        blue_score, var_blue_score = evaluator.blue_score()
        evaluator.infer_predictions()
        conf_score = evaluator.get_conf_score_pred()
        lf_score = evaluator.lf_score()

        o_str = f"{fold}\t{cfg_gen}\t{blue_score}\t{var_blue_score}\t{lf_score}\t{conf_score}\n"
        o_file.write(o_str)
