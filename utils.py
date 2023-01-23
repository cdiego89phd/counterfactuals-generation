import transformers
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from openprompt.data_utils import InputExample
import bs4
import pynvml


def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    print("########################################")


def load_dataset(loading_path):
    train = pd.read_csv(loading_path + "training_set.tsv", sep='\t')
    val = pd.read_csv(loading_path + "val_set.tsv", sep='\t')
    test = pd.read_csv(loading_path + "test_set.tsv", sep='\t')
    return train, val, test


def load_dataset_with_val(seed, val_prop, loading_path):
    data = pd.read_csv(loading_path, sep='\t')

    # split into train val
    valset = data.sample(frac=val_prop, replace=False, random_state=seed)
    trainset = data[~data.index.isin(valset.index)]

    return trainset, valset


def wrap_dataset_with_prompt(df_row, template, mapping_labels, spec_tokens):
    final_text = template.replace("<label_ex>", mapping_labels[df_row["label_ex"]])
    final_text = final_text.replace("<example_text>", df_row["example"])
    final_text = final_text.replace("<label_counter>", mapping_labels[df_row["label_counter"]])
    if "<counter_text>" in final_text:
        final_text = final_text.replace("<counter_text>", df_row["counterfactual"])

    if spec_tokens != "None":
        final_text = final_text.replace("<sep>", spec_tokens["sep_token"])
        final_text = final_text.replace("<bos_token>", spec_tokens["bos_token"])
        final_text = final_text.replace("<eos_token>", spec_tokens["eos_token"])

    # this is for vanilla generation
    # we create a prompt with some words from the seed review
    words = df_row["example"].split(" ")
    final_text = final_text.replace("<0>", words[0])
    final_text = final_text.replace("<1>", words[1])
    final_text = final_text.replace("<2>", words[2])

    return final_text


def wrap_nli_dataset_with_prompt(df_row, template, spec_tokens):
    final_text = template.replace("<original_label>", df_row["original_label"])
    final_text = final_text.replace("<P>", df_row["original_prem"])
    final_text = final_text.replace("<H>", df_row["original_hyp"])
    final_text = final_text.replace("<counter_label>", df_row["counter_label"])
    final_text = final_text.replace("<TC>", df_row["task"])

    if df_row["task"] == "RP":
        counter_text = df_row["counter_prem"]
    else:
        counter_text = df_row["counter_hyp"]
    final_text = final_text.replace("<counter_text>", counter_text)

    if spec_tokens != "None":
        final_text = final_text.replace("<sep>", spec_tokens["sep_token"])
        final_text = final_text.replace("<bos_token>", spec_tokens["bos_token"])
        final_text = final_text.replace("<eos_token>", spec_tokens["eos_token"])

    return final_text


def load_tokenizer(tok_name, spec_tokens="None") -> transformers.AutoTokenizer:
    tok = transformers.AutoTokenizer.from_pretrained(tok_name)
    print("Downloaded tokenizer!")
    if spec_tokens != "None":
        print(f"Len of tokenizer before adding tokens:{len(tok)}")
        tok.add_special_tokens(spec_tokens)  # add special tokens
        print("Added special tokens to tokenizer!")
        print(f"Len of tokenizer after adding tokens:{len(tok)}")
    return tok


def load_causal_model(model_name: str, n_tokens: int, spec_tokens="None") -> \
        (transformers.AutoModelForCausalLM, transformers.AutoConfig):
    model_config_class = transformers.AutoConfig.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                              load_in_8bit=True,
                                                              device_map='sequential')
    # model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    print("Downloaded model and cfg!")
    if spec_tokens != "None":
        # special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(n_tokens)
    return model, model_config_class


def load_causal_model_from_local(model_path):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    return model

# def load_gpt2_objects(model_name, spec_tokens):
#     # Load language gpt2 objects
#     tok = transformers.GPT2Tokenizer.from_pretrained(model_name)
#     print("Downloaded tokenizer!")
#     if spec_tokens != "None":
#         print(f"Len of tokenizer before adding tokens:{len(tok)}")
#         tok.add_special_tokens(spec_tokens)  # add special tokens
#         print("Added special tokens to tokenizer!")
#         print(f"Len of tokenizer after adding tokens:{len(tok)}")
#
#     model_config_class = transformers.GPT2Config.from_pretrained(model_name)
#     model = transformers.GPT2LMHeadModel.from_pretrained(model_name, config=model_config_class)
#     print("Downloaded model and cfg!")
#     if spec_tokens != "None":
#         # special tokens added, model needs to be resized accordingly
#         model.resize_token_embeddings(len(tok))
#
#     return tok, model, model_config_class


# def load_gpt2_from_local(model_path):
#     model = transformers.GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
#     return model


def prepare_sentiment_classifier(classifier_name):
    # load the sentiment classifier
    classifier_tokenizer = transformers.AutoTokenizer.from_pretrained(classifier_name)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_name)

    # TODO in the future, check this dict is the one used by the classifier
    # there will be different dicts for different classifiers
    classifier_label_map = {'NEGATIVE':  0, 'POSITIVE': 1}

    classification_tools = {"tokenizer": classifier_tokenizer,
                            "classifier": classifier,
                            "label_map": classifier_label_map}

    return classification_tools


def prepare_nli_classifier(classifier_name):
    # cross-encoder/nli-deberta-v3-large is the best classifier so far
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_name)
    classifier_tokenizer = transformers.AutoTokenizer.from_pretrained(classifier_name)

    if classifier_name in ["ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                           "pepa/bigbird-roberta-large-snli",
                           "textattack/albert-base-v2-snli",
                           "textattack/distilbert-base-cased-snli"]:
        classifier_label_map = {"entailment": 0,
                                "neutral": 1,
                                "contradiction": 2
                                }
    else:
        classifier_label_map = {"contradiction": 0,
                                "entailment": 1,
                                "neutral": 2
                                }

    classification_tools = {"tokenizer": classifier_tokenizer,
                            "classifier": classifier,
                            "label_map": classifier_label_map}
    return classification_tools


def generate_batches(data, n):
    batch_size = len(data)//n
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
    return data


@dataclass
class TaskDataset(Dataset):
    raw_dataframe: pd.DataFrame

    def __init__(self, raw_dataframe):
        super(Dataset, self).__init__()
        self.raw_dataframe = raw_dataframe
        self.guids = []
        self.dataset = {}

    def __getitem__(self, idx) -> InputExample:
        """Return the item of index idx """
        return self.dataset[idx]

    def __len__(self) -> int:
        """Return len of dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return iterator of the dataset. Implemented because of inheritance from Dataset"""
        return iter(self.dataset)

    def __next__(self):
        """Return next item of dataset"""
        return iter(self.dataset)

    def get_dataset(self) -> dict:
        """Return the dataset in Dataset format (dict of InputExample)"""
        return self.dataset

    def get_raw_dataframe(self) -> pd.DataFrame:
        """Return the raw dataset in pandas format"""
        return self.raw_dataframe


@dataclass
class SentimentDataset(TaskDataset):
    def __init__(self, raw_dataframe):
        super().__init__(raw_dataframe)

    def prepare_dataloader(self) -> None:
        """Convert the raw_dataframe into the InputExample format dataset of openprompt
        """
        for index, row in self.raw_dataframe.iterrows():
            self.dataset[row['paired_id']] = InputExample(guid=row['paired_id'],
                                                          text_a=bs4.BeautifulSoup(
                                                              row['wrapped_input'], "lxml").text,
                                                          meta={"label_ex": row['label_ex'],
                                                                "label_counter": row['label_counter'],
                                                                'example': bs4.BeautifulSoup(
                                                                    row['example'], "lxml").text,
                                                                'counterfactual': bs4.BeautifulSoup(
                                                                    row['counterfactual'], "lxml").text})
            self.guids.append(row['paired_id'])
        print('Dataloader prepared!')

    @staticmethod
    def to_dataframe(n_to_generate, dataset: TaskDataset) -> pd.DataFrame:
        """Build a dataframe from dataset of InputExample"""

        paired_ids = [idx for idx in dataset]
        labels_ex = [dataset.__getitem__(idx).meta["label_ex"] for idx in dataset]
        examples = [dataset.__getitem__(idx).meta["example"] for idx in dataset]
        labels_counter = [dataset.__getitem__(idx).meta["label_counter"] for idx in dataset]
        counterfactuals = [dataset.__getitem__(idx).meta["counterfactual"] for idx in dataset]
        generated_counters = [dataset.__getitem__(idx).meta["generated_counter"] for idx in dataset]
        d = {"paired_id": paired_ids,
             "label_ex": labels_ex,
             "example": examples,
             "label_counter": labels_counter,
             "counterfactual": counterfactuals,
             }
        for idx in range(n_to_generate):
            d[f"generated_counter_{idx}"] = []

        for item in generated_counters:
            for idx in range(len(item)):
                d[f"generated_counter_{idx}"].append(item[idx])

        return pd.DataFrame(data=d)


@dataclass
class NLIDataset(TaskDataset):
    def __init__(self, raw_dataframe):
        super().__init__(raw_dataframe)

    @staticmethod
    def check_nan(text):
        if str(text) == 'nan':
            return "None"
        else:
            return text

    def prepare_dataloader(self) -> None:
        """Convert the raw_dataframe into the InputExample format dataset of openprompt
        """
        for index, row in self.raw_dataframe.iterrows():
            self.dataset[index] = InputExample(guid=index,
                                               text_a=bs4.BeautifulSoup(
                                                   row['wrapped_input'], "lxml").text,
                                               meta={"original_label": row['original_label'],
                                                     "counter_label": row['counter_label'],
                                                     "task": row['task'],
                                                     'original_prem': bs4.BeautifulSoup(
                                                         self.check_nan(row['original_prem']), "lxml").text,
                                                     'counter_prem': bs4.BeautifulSoup(
                                                         self.check_nan(row['counter_prem']), "lxml").text,
                                                     'original_hyp': bs4.BeautifulSoup(
                                                         self.check_nan(row['original_hyp']), "lxml").text,
                                                     'counter_hyp': bs4.BeautifulSoup(
                                                         self.check_nan(row['counter_hyp']), "lxml").text})
            self.guids.append(index)
        print('Dataloader prepared!')

    @staticmethod
    def to_dataframe(n_to_generate: int, dataset: TaskDataset) -> pd.DataFrame:
        """Build a dataframe from dataset of InputExample"""

        ids = [idx for idx in dataset]
        original_labels = [dataset.__getitem__(idx).meta["original_label"] for idx in dataset]
        counter_labels = [dataset.__getitem__(idx).meta["counter_label"] for idx in dataset]
        tasks = [dataset.__getitem__(idx).meta["task"] for idx in dataset]
        original_prems = [dataset.__getitem__(idx).meta["original_prem"] for idx in dataset]
        counter_prems = [dataset.__getitem__(idx).meta["counter_prem"] for idx in dataset]
        original_hyps = [dataset.__getitem__(idx).meta["original_hyp"] for idx in dataset]
        counter_hyps = [dataset.__getitem__(idx).meta["counter_hyp"] for idx in dataset]
        generated_counters = [dataset.__getitem__(idx).meta["generated_counter"] for idx in dataset]

        d = {"id": ids,
             "original_label": original_labels,
             "counter_label": counter_labels,
             "task": tasks,
             "original_prem": original_prems,
             "counter_prem": counter_prems,
             "original_hyp": original_hyps,
             "counter_hyp": counter_hyps,
             }
        for idx in range(n_to_generate):
            d[f"generated_counter_{idx}"] = []

        for item in generated_counters:
            for idx in range(len(item)):
                d[f"generated_counter_{idx}"].append(item[idx])

        return pd.DataFrame(data=d)
