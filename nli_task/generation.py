import pandas as pd
import torch
from torch.utils.data import Dataset
import datetime
import bs4
import math
from dataclasses import dataclass, field

import openprompt
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample

import traceback


@dataclass
class NLIDataset(Dataset):
    raw_dataframe: pd.DataFrame
    guids: list = field(default_factory=list)
    dataset: dict = field(default_factory=dict)

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
            self.dataset[row['paired_id']] = InputExample(guid=index,
                                                          text_a=bs4.BeautifulSoup(
                                                              row['wrapped_input'], "lxml").text,
                                                          meta={"original_label": row['original_label'],
                                                                "counter_label": row['counter_label'],
                                                                "task": row['task'],
                                                                'original_prem': bs4.BeautifulSoup(
                                                                    self.check_nan(row['original_prem']), "lxml").text,
                                                                'counter_prem': bs4.BeautifulSoup(
                                                                    self.check_nan(row['counter_prem']), "lxml").text})
            self.guids.append(index)
    print('Dataloader prepared!')

    def get_dataset(self) -> dict:
        """Return the dataset in Dataset format (dict of InputExample)"""
        return self.dataset

    def get_raw_dataframe(self) -> pd.DataFrame:
        """Return the raw dataset in pandas format"""
        return self.raw_dataframe


class CounterGenerator:
    def __init__(self,
                 template: ManualTemplate,
                 lm,
                 dataloader: openprompt.PromptDataLoader,
                 dataset: NLIDataset,
                 cfgs: dict):
        """Constructor of the counterfactual generator
        @param: dataloader That store the dataset
        @param: dataset TODO
        @param: generator The generator TODO
        @param: Generation params TODO
        """
        self.dataloader = dataloader
        self.dataset = dataset
        self.gen_cfgs = cfgs

        self.generator = openprompt.PromptForGeneration(
            template=template,
            freeze_plm=True,
            plm=lm,
            plm_eval_mode=True
        )

        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
        print(f"The counterfactual generator is placed on cuda device:{self.generator.device.index}")

    def perform_generation(self, tokenizer, n_to_generate=1):
        print(f"# to generate:{n_to_generate}")
        self.generator.eval()

        for (step, inputs) in enumerate(self.dataloader):

            # retrieve the instance involved
            instance_guid = inputs["guid"].numpy()[0]
            instance_to_update = self.dataset.__getitem__(instance_guid)

            # we limit the output length to be reasonably equal to the input
            # context, i.e. the example
            max_length_example = len(tokenizer.encode(instance_to_update.text_a))
            max_length_output = int(2 * max_length_example)
            if max_length_output > 1024:
                max_length_output = 1024

            generation_arguments = {
                "max_length": max_length_output,
                "min_length": 5,
                "no_repeat_ngram_size": self.gen_cfgs["no_repeat_ngram_size"],
                "num_beams": self.gen_cfgs["num_beams"],
                "repetition_penalty": float(self.gen_cfgs["repetition_penalty"]),
                "temperature": float(self.gen_cfgs["temperature"]),
                "do_sample": self.gen_cfgs["do_sample"],
                "num_return_sequences": 1,
                "top_k": self.gen_cfgs["top_k"],
                "top_p": self.gen_cfgs["top_p"],
            }

            try:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()

                instance_to_update.meta["generated_counter"] = []
                for i in range(n_to_generate):
                    _, generated_counter = self.generator.generate(inputs,
                                                                   verbose=False,
                                                                   **generation_arguments)
                    instance_to_update.meta["generated_counter"].append(generated_counter[0])

            except Exception as e:
                instance_to_update.meta["generated_counter"] = None
                print(f"Instance that generated the exception:{instance_guid}")
                print(f"Step:{step}")
                print(e)
                print(traceback.format_exc())

            if (step % 100) == 0 and (step > 0):
                print(f"{datetime.datetime.now()}, Step:{step}: 100 counterfactuals generated")

    def dataframe_from_dataset(self, n_to_generate) -> pd.DataFrame:
        """Build a dataframe from dataset"""

        paired_ids = [idx for idx in self.dataset]
        labels_ex = [self.dataset.__getitem__(idx).meta["label_ex"] for idx in self.dataset]
        examples = [self.dataset.__getitem__(idx).meta["example"] for idx in self.dataset]
        labels_counter = [self.dataset.__getitem__(idx).meta["label_counter"] for idx in self.dataset]
        counterfactuals = [self.dataset.__getitem__(idx).meta["counterfactual"] for idx in self.dataset]
        generated_counters = [self.dataset.__getitem__(idx).meta["generated_counter"] for idx in self.dataset]
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
