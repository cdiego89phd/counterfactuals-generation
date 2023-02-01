import pandas as pd
import transformers
import torch
import numpy as np
import sklearn
import nltk
import stanza
import zss
import scipy.stats
import fast_bleu

import utils


class Evaluator(object):
    def __init__(self, tokenizer, model, eval_dataset):
        self.tokenizer = tokenizer
        self.model = model
        self.df_eval_dataset = eval_dataset

    def clean_evalset(self) -> int:
        """Remove some nan values in the generated counterfactuals"""

        n_nan = self.df_eval_dataset['generated_counter_0'].isna().sum()
        print(f"# of nan values removed in generated counterfactuals:{n_nan}")
        self.df_eval_dataset.dropna(subset=['generated_counter_0'], inplace=True)
        self.df_eval_dataset.dropna(subset=['generated_counter_1'], inplace=True)
        self.df_eval_dataset.dropna(subset=['generated_counter_2'], inplace=True)

        return n_nan

    @staticmethod
    def text_size_words(text: str):
        return len(nltk.tokenize.word_tokenize(text))

    def counter_sizes(self, row: pd.Series, n: int):
        return np.mean([self.text_size_words(
            row[f"generated_counter_{i}"]) for i in range(n)])

    def calculate_sizes(self, n_counter_generated: int) -> None:
        """Calculate the avg sizes of the counterfactuals (both annotated and generated)"""

        self.df_eval_dataset["counter_size"] = \
            [self.text_size_words(text) for text in self.df_eval_dataset["counterfactual"]]

        self.df_eval_dataset["generated_counter_size"] = self.df_eval_dataset \
            .apply(lambda row: self.counter_sizes(row, n_counter_generated), axis=1)

    def get_eval_set(self):
        return self.df_eval_dataset

    @staticmethod
    def calculate_correlation(x_values, y_values) -> (float, float):
        spear, _ = scipy.stats.spearmanr(x_values, y_values)
        pears, _ = scipy.stats.pearsonr(x_values, y_values)
        return spear, pears

    def calculate_bleu_corpus(self, n_generated: int,
                              weights=(0.25, 0.25, 0.25, 0.25)) -> float:
        """Calculate the corpus BLUE score for the entire set of reference-hypothesis pairs.
           Returns the BLUE score.
        """
        true_counters = self.df_eval_dataset["counterfactual"].values
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
        scores = []
        for idx in range(n_generated):
            gen_counters = self.df_eval_dataset[f"generated_counter_{idx}"].values

            # need to tokenize sentences
            # the reference is the true counterfactual
            references = [[nltk.tokenize.word_tokenize(sentence)] for sentence in true_counters]

            # the hypothesis is the generated counterfactual
            hypothesis = [nltk.tokenize.word_tokenize(sentence) for sentence in gen_counters]

            score = nltk.translate.bleu_score.corpus_bleu(references,
                                                          hypothesis,
                                                          weights=weights,
                                                          smoothing_function=smoothing_function.method1)
            scores.append(score)

        return float(np.mean(scores))

    def calculate_bleu_score(self, n_generated: int,
                             weights=(0.25, 0.25, 0.25, 0.25),
                             calculate_corr=False) -> (float, float, float, float):
        """Calculate the BLEU score for a pair of example-counter.
           Returns mean and variance of the BLEU scores.
        """

        true_counters = self.df_eval_dataset["counterfactual"].values
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
        mean_scores = []
        var_scores = []
        spear_scores = []
        pears_scores = []
        for idx in range(n_generated):
            blue_score = []
            gen_counters = self.df_eval_dataset[f"generated_counter_{idx}"].values
            for true_counter, gen_counter in zip(true_counters, gen_counters):
                # example and counterfactual need to be tokenized first

                # the reference is the true counterfactual
                reference = nltk.tokenize.word_tokenize(true_counter)

                # the hypothesis is the generated counterfactual
                hypothesis = nltk.tokenize.word_tokenize(gen_counter)

                blue_score.append(
                    nltk.translate.bleu_score.sentence_bleu([reference],
                                                            hypothesis,
                                                            weights=weights,
                                                            smoothing_function=smoothing_function.method1))

            if calculate_corr:
                spear_corr, pears_corr = \
                    self.calculate_correlation(blue_score, self.df_eval_dataset["counter_size"].values)
                spear_scores.append(spear_corr)
                pears_scores.append(pears_corr)

            mean_scores.append(np.mean(blue_score))
            var_scores.append(np.var(blue_score))

        if calculate_corr:
            return np.mean(mean_scores), np.var(var_scores), np.mean(spear_scores), np.mean(pears_scores)
        else:
            return np.mean(mean_scores), np.var(var_scores), -100, -100  # default values for correlation

    def calculate_lev_dist(self, n_generated: int, calculate_corr=False) -> (float, float, float, float):
        """Calculate the Levenshtein Distance score for the entire set of reference-hypothesis pairs.
           Returns the Levenshtein Distance score.
        """

        mean_scores = []
        var_scores = []
        spear_scores = []
        pears_scores = []

        if type(self) == NLIEvaluator:
            original_examples = \
                self.df_eval_dataset.apply(lambda row: NLIEvaluator.extract_original_text(row), axis=1).values
        else:
            original_examples = self.df_eval_dataset["example"].values

        for idx in range(n_generated):
            distances = [nltk.edit_distance(nltk.tokenize.word_tokenize(str_1),
                                            nltk.tokenize.word_tokenize(str_2))/len(nltk.tokenize.word_tokenize(str_1))
                         for (str_1, str_2) in zip(original_examples,
                                                   self.df_eval_dataset[f"generated_counter_{idx}"].values)]
            mean_scores.append(np.mean(distances))
            var_scores.append(np.var(distances))

            if calculate_corr:
                spear_corr, pears_corr = \
                    self.calculate_correlation(distances, self.df_eval_dataset["counter_size"].values)
                spear_scores.append(spear_corr)
                pears_scores.append(pears_corr)

        if calculate_corr:
            return np.mean(mean_scores), np.var(var_scores), np.mean(spear_scores), np.mean(pears_scores)
        else:
            return np.mean(mean_scores), np.var(var_scores), -100, -100  # default values for correlation

    @staticmethod
    def adjust_trees_and_calculate(pipeline, tree_a, tree_b) -> float:
        # check and add empty sentences
        if len(tree_a.sentences) >= len(tree_b.sentences):
            n_to_add = len(tree_a.sentences) - len(tree_b.sentences)
            to_add = [pipeline(".").sentences[0] for i in range(n_to_add)]
            tree_b.sentences += to_add
        else:
            n_to_add = len(tree_b.sentences) - len(tree_a.sentences)
            to_add = [pipeline(".").sentences[0] for i in range(n_to_add)]
            tree_a.sentences += to_add

        assert len(tree_a.sentences) == len(tree_b.sentences)

        distances = [zss.simple_distance(s_one.constituency, s_two.constituency) for (s_one, s_two)
                     in zip(tree_a.sentences, tree_b.sentences)]

        return float(np.mean(distances))

    def calculate_zss_dist(self, n_generated: int, calculate_corr=False) -> (float, float, float, float):
        """Calculate tree-edit distance (by Zhang and Sasha 1989) for the entire set of reference-hypothesis pairs.
           Tree-edit distance is defined as the minimum number of transformations required to turn the constituency
           parse tree of a generated counterfactual review to that of the reference counterfactual review.
           Returns the tree-edit distance score.
        """
        pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

        # parse reviews into constituency trees
        if type(self) == NLIEvaluator:
            original_examples = \
                self.df_eval_dataset.apply(lambda row: NLIEvaluator.extract_original_text(row), axis=1).values
        else:
            original_examples = self.df_eval_dataset["example"].values
        parsed_references = [pipeline(review) for review in original_examples]

        mean_scores = []
        var_scores = []
        spear_scores = []
        pears_scores = []
        for idx in range(n_generated):
            parsed_generated = \
                [pipeline(review) for review in self.df_eval_dataset[f"generated_counter_{idx}"].values]
            distances = [self.adjust_trees_and_calculate(pipeline, tree_a, tree_b) for (tree_a, tree_b)
                         in zip(parsed_references, parsed_generated)]
            mean_scores.append(np.mean(distances))
            var_scores.append(np.var(distances))

            if calculate_corr:
                spear_corr, pears_corr = \
                    self.calculate_correlation(distances, self.df_eval_dataset["counter_size"].values)
                spear_scores.append(spear_corr)
                pears_scores.append(pears_corr)

        if calculate_corr:
            return np.mean(mean_scores), np.var(var_scores), np.mean(spear_scores), np.mean(pears_scores)
        else:
            return np.mean(mean_scores), np.var(var_scores), -100, -100  # default values for correlation

    def calculate_self_bleu(self, n_generated,
                            weights=None,
                            calculate_corr=False) -> (float, float, float, float):
        """Calculate the self-BLEU score for the n_generated counterfactuals.
           Returns mean and variance of the self-BLEU scores.
        """
        if weights is None:
            weights = {'4_gram': (1 / 4., 1 / 4., 1 / 4., 1 / 4.)}
        scores = []

        # iterate over rows of the eval_dataset
        for index, row in self.df_eval_dataset.iterrows():

            # retrieve and tokenize counterfactuals
            references = [nltk.tokenize.word_tokenize(row[f"generated_counter_{idx}"])
                          for idx in range(n_generated)]

            self_bleu = fast_bleu.SelfBLEU(references, weights)
            scores.append(np.mean(self_bleu.get_score()['4_gram']))

        if calculate_corr:
            spear_corr, pears_corr = \
                self.calculate_correlation(scores, self.df_eval_dataset["counter_size"].values)
            return np.mean(scores), np.var(scores), spear_corr, pears_corr
        else:
            return np.mean(scores), np.var(scores), -100, -100  # default values for correlation


class SentimentEvaluator(Evaluator):

    def __init__(self, tokenizer, model, label_map, eval_dataset, s_device=0):
        super().__init__(tokenizer, model, eval_dataset)

        self.classifier = transformers.pipeline(
            task="sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=s_device)

        self.label_map = label_map
        self.score_labels = {}
        self.predicted_labels = {}

    def infer_predictions(self, n_generated=1):
        """Infer the labels for the counterfactuals"""

        for idx in range(n_generated):
            texts = self.df_eval_dataset[f"generated_counter_{idx}"].values
            self.predicted_labels[idx] = []
            self.score_labels[idx] = []

            for text_to_classify in texts:
                if len(text_to_classify) > 512:
                    result = self.classifier(text_to_classify[:511])[0]
                else:
                    result = self.classifier(text_to_classify)[0]

                self.predicted_labels[idx].append(self.label_map[result['label']])
                self.score_labels[idx].append(result['score'])

    def calculate_lf_score(self) -> float:
        """Calculate the Label Flip Score (LFS)"""

        y_desired = self.df_eval_dataset["label_counter"].values

        scores = []
        for idx in self.predicted_labels.keys():
            scores.append(sklearn.metrics.accuracy_score(y_desired, self.predicted_labels[idx]))
        return float(np.mean(scores))

    def get_conf_score_pred(self) -> float:
        scores = []
        for idx in self.score_labels.keys():
            scores.append(np.mean(self.score_labels[idx]))
        return float(np.mean(scores))


class NLIEvaluator(Evaluator):
    def __init__(self, tokenizer, model, label_map, eval_dataset):
        super().__init__(tokenizer, model, eval_dataset)

        self.df_eval_dataset["counterfactual"] = \
            self.df_eval_dataset.apply(lambda row: self.extract_counter_text(row), axis=1)

        self.model.cuda()
        self.label_map = label_map
        self.batches = {}
        self.predicted_labels = {}

    @staticmethod
    def extract_prems(row: pd.Series, generated_label: str) -> str:
        if row["task"] == "RP":
            return row[generated_label]
        else:
            return row["original_prem"]

    @staticmethod
    def extract_hyps(row: pd.Series, generated_label: str) -> str:
        if row["task"] == "RH":
            return row[generated_label]
        else:
            return row["original_hyp"]

    @staticmethod
    def extract_counter_text(row) -> str:
        if row["task"] == "RH":
            return row["counter_hyp"]
        else:
            return row["counter_prem"]

    @staticmethod
    def extract_original_text(row) -> str:
        if row["task"] == "RP":
            return row["original_prem"]
        else:
            return row["original_hyp"]

    def prepare_batches(self, n_batches: int, n_generated=1) -> None:
        for i in range(n_generated):
            gen_label = f"generated_counter_{i}"
            premises = self.df_eval_dataset.apply(lambda row: self.extract_prems(row, gen_label), axis=1).values
            hypos = self.df_eval_dataset.apply(lambda row: self.extract_hyps(row, gen_label), axis=1).values
            eval_batch = [[p, h] for p, h in zip(premises, hypos)]
            self.batches[f"generated_counter_{i}"] = utils.generate_batches(eval_batch, n_batches)

    def infer_predictions(self, n_generated=1) -> None:
        """Infer the labels for the counterfactuals"""

        self.model.eval()
        # we have n_generated set of batches to predict
        with torch.no_grad():
            for i in range(n_generated):
                batches = self.batches[f"generated_counter_{i}"]

                predictions = []
                for batch in batches:
                    tokenized_batch = self.tokenizer(batch,
                                                     padding=True,
                                                     truncation=True,
                                                     return_tensors="pt")
                    scores = self.model(**tokenized_batch.to('cuda')).logits
                    predictions += [score_max.cpu().numpy() for score_max in scores.argmax(dim=1)]
                self.predicted_labels[f"generated_counter_{i}"] = predictions

    def calculate_lf_score(self) -> float:
        """It corresponds to the accuracy of the classifier with respect to the
           true conterfactual labels
        """
        y_desired = [self.label_map[label] for label in self.df_eval_dataset["counter_label"].values]

        scores = []
        for idx in self.predicted_labels.keys():
            scores.append(sklearn.metrics.accuracy_score(y_desired, self.predicted_labels[idx]))
        return float(np.mean(scores))
