import pandas as pd
import transformers
import numpy as np
import sklearn
import nltk
# import Levenshtein
import stanza
import zss
import scipy.stats
import fast_bleu


class SentimentEvaluator:

    def __init__(self, s_tokenizer, s_model, s_label_dict, s_device=0):
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

        self.predicted_labels = {}
        self.score_labels = {}

    @staticmethod
    def clean_evalset(eval_dataset) -> (pd.DataFrame, int):
        """Remove some nan values in the generated counterfactuals"""

        n_nan = eval_dataset['generated_counter_0'].isna().sum()
        print(f"# of nan values removed in generated counterfactuals:{n_nan}")
        return eval_dataset.dropna(), n_nan

    @staticmethod
    def text_size_words(text):
        return len(nltk.tokenize.word_tokenize(text))

    @staticmethod
    def counter_sizes(row, n):
        return np.mean([SentimentEvaluator.text_size_words(row[f"generated_counter_{i}"]) for i in range(n)])

    @staticmethod
    def retrieve_sizes(eval_dataset, n_counter_generated) -> pd.DataFrame:
        """Calculate the avg sizes of the counterfactual reviews"""

        # jdsjdl
        eval_dataset["counter_size"] = [SentimentEvaluator.text_size_words(row) for row in
                                        eval_dataset['counterfactual'].values]
        eval_dataset["generated_counter_size"] = eval_dataset\
            .apply(lambda row: SentimentEvaluator.counter_sizes(row, n_counter_generated), axis=1)
        return eval_dataset

    def infer_predictions(self, eval_dataset, n_generated=1):
        """Infer the labels for the counterfactuals"""

        for idx in range(n_generated):
            texts = eval_dataset[f"generated_counter_{idx}"].values
            self.predicted_labels[idx] = []
            self.score_labels[idx] = []

            for text_to_classify in texts:
                if len(text_to_classify) > 512:
                    result = self.classifier(text_to_classify[:511])[0]
                else:
                    result = self.classifier(text_to_classify)[0]

                self.predicted_labels[idx].append(self.s_label_dict[result['label']])
                self.score_labels[idx].append(result['score'])

    def calculate_lf_score(self, eval_dataset) -> float:
        """Calculate the Label Flip Score (LFS)"""

        y_desired = eval_dataset["label_counter"].values

        scores = []
        for idx in self.predicted_labels.keys():
            scores.append(sklearn.metrics.accuracy_score(y_desired, self.predicted_labels[idx]))
        return np.mean(scores)

    def get_conf_score_pred(self) -> float:
        scores = []
        for idx in self.score_labels.keys():
            scores.append(np.mean(self.score_labels[idx]))
        return np.mean(scores)

    @staticmethod
    def calculate_correlation(eval_dataset, x_values) -> (float, float):

        counter_sizes = eval_dataset['counter_size'].values
        spear, _ = scipy.stats.spearmanr(x_values, counter_sizes)
        pears, _ = scipy.stats.pearsonr(x_values, counter_sizes)
        return spear, pears

    @staticmethod
    def calculate_bleu_score(eval_dataset, n_generated,
                             weights=(0.25, 0.25, 0.25, 0.25),
                             calculate_corr=False) -> (float, float, float, float):
        """Calculate the BLEU score for a pair of example-counter.
           Returns mean and variance of the BLEU scores.
        """
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
        true_counters = eval_dataset["counterfactual"].values

        mean_scores = []
        var_scores = []
        spear_scores = []
        pears_scores = []
        for idx in range(n_generated):
            blue_score = []
            gen_counters = eval_dataset[f"generated_counter_{idx}"].values
            for true_counter, gen_counter in zip(true_counters, gen_counters):
                # example and counterfactual need to be tokenized first

                # the reference is the true counterfactual
                reference = nltk.tokenize.word_tokenize(true_counter)

                # the hypothesis is the generated counterfactual
                hypothesis = nltk.tokenize.word_tokenize(gen_counter)

                blue_score.append(nltk.translate.bleu_score.sentence_bleu([reference],
                                                                          hypothesis,
                                                                          weights=weights,
                                                                          smoothing_function=smoothing_function.method1))

            if calculate_corr:
                spear_corr, pears_corr = SentimentEvaluator.calculate_correlation(eval_dataset, blue_score)
                spear_scores.append(spear_corr)
                pears_scores.append(pears_corr)

            mean_scores.append(np.mean(blue_score))
            var_scores.append(np.var(blue_score))

        if calculate_corr:
            return np.mean(mean_scores), np.var(var_scores), np.mean(spear_scores), np.mean(pears_scores)
        else:
            return np.mean(mean_scores), np.var(var_scores), -100, -100  # default values for correlation

    @staticmethod
    def calculate_bleu_corpus(eval_dataset, n_generated,
                              weights=(0.25, 0.25, 0.25, 0.25)) -> float:
        """Calculate the corpus BLUE score for the entire set of reference-hypothesis pairs.
           Returns the BLUE score.
        """
        smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
        true_counters = eval_dataset["counterfactual"].values

        scores = []
        for idx in range(n_generated):
            gen_counters = eval_dataset[f"generated_counter_{idx}"].values

            # need to tokenize sentences
            # the reference is the true counterfactual
            references = [nltk.tokenize.word_tokenize(sentence) for sentence in true_counters]

            # the hypothesis is the generated counterfactual
            hypothesis = [nltk.tokenize.word_tokenize(sentence) for sentence in gen_counters]

            score = nltk.translate.bleu_score.corpus_bleu(references,
                                                          hypothesis,
                                                          weights=weights,
                                                          smoothing_function=smoothing_function.method1)
            scores.append(score)

        return np.mean(scores)

    @staticmethod
    def calculate_self_bleu(eval_dataset, n_generated,
                            weights=None,
                            calculate_corr=False) -> (float, float, float, float):
        """Calculate the self-BLEU score for the n_generated counterfactuals.
           Returns mean and variance of the self-BLEU scores.
        """
        if weights is None:
            weights = {'4_gram': (1 / 4., 1 / 4., 1 / 4., 1 / 4.)}
        scores = []

        # iterate over rows of the eval_dataset
        for index, row in eval_dataset.iterrows():

            # retrieve and tokenize counterfactuals
            references = [nltk.tokenize.word_tokenize(row[f"generated_counter_{idx}"])
                          for idx in range(n_generated)]

            self_bleu = fast_bleu.SelfBLEU(references, weights)
            scores.append(np.mean(self_bleu.get_score()['4_gram']))

        if calculate_corr:
            spear_corr, pears_corr = SentimentEvaluator.calculate_correlation(eval_dataset, scores)
            return np.mean(scores), np.var(scores), spear_corr, pears_corr
        else:
            return np.mean(scores), np.var(scores), -100, -100  # default values for correlation

    @staticmethod
    def calculate_lev_dist(eval_dataset, n_generated, calculate_corr=False) -> (float, float, float, float):
        """Calculate the Levenshtein Distance score for the entire set of reference-hypothesis pairs.
           Returns the Levenshtein Distance score.
        """

        mean_scores = []
        var_scores = []
        spear_scores = []
        pears_scores = []
        for idx in range(n_generated):
            # nltk.edit_distance(tokenized_original, tokenized_edited)
            distances = [nltk.edit_distance(nltk.tokenize.word_tokenize(str_1),
                                            nltk.tokenize.word_tokenize(str_2))/len(nltk.tokenize.word_tokenize(str_1))
                         for (str_1, str_2) in zip(eval_dataset["example"].values,
                                                   eval_dataset[f"generated_counter_{idx}"].values)]
            # distances = [Levenshtein.distance(str_1, str_2)/len(str_1) for (str_1, str_2) in
            #              zip(eval_dataset["example"].values, eval_dataset[f"generated_counter_{idx}"].values)]
            mean_scores.append(np.mean(distances))
            var_scores.append(np.var(distances))

            if calculate_corr:
                spear_corr, pears_corr = SentimentEvaluator.calculate_correlation(eval_dataset, distances)
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

        return np.mean(distances)

    @staticmethod
    def calculate_zss_dist(eval_dataset, n_generated, calculate_corr=False) -> (float, float, float, float):
        """Calculate tree-edit distance (by Zhang and Sasha 1989) for the entire set of reference-hypothesis pairs.
           Tree-edit distance is defined as the minimum number of transformations required to turn the constituency
           parse tree of a generated counterfactual review to that of the reference counterfactual review.
           Returns the tree-edit distance score.
        """
        pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

        # parse reviews into constituency trees
        parsed_references = [pipeline(review) for review in eval_dataset["example"].values]

        mean_scores = []
        var_scores = []
        spear_scores = []
        pears_scores = []
        for idx in range(n_generated):
            parsed_generated = [pipeline(review) for review in eval_dataset[f"generated_counter_{idx}"].values]
            distances = [SentimentEvaluator.adjust_trees_and_calculate(pipeline, tree_a, tree_b) for (tree_a, tree_b)
                         in zip(parsed_references, parsed_generated)]
            mean_scores.append(np.mean(distances))
            var_scores.append(np.var(distances))

            if calculate_corr:
                spear_corr, pears_corr = SentimentEvaluator.calculate_correlation(eval_dataset, distances)
                spear_scores.append(spear_corr)
                pears_scores.append(pears_corr)

        if calculate_corr:
            return np.mean(mean_scores), np.var(var_scores), np.mean(spear_scores), np.mean(pears_scores)
        else:
            return np.mean(mean_scores), np.var(var_scores), -100, -100  # default values for correlation
