import pandas as pd
import torch
from torch.utils.data import Dataset
import datetime
import bs4
from dataclasses import dataclass, field

import openprompt
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample


@dataclass
class SentimentDataset(Dataset):
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
                 dataset: SentimentDataset,
                 cuda_device: torch.device,
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
        print(f"index: {self.generator.device.index}")

        print("DIEGO")
        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            # with torch.cuda.device(1):
            #     self.generator = self.generator.to(cuda_device).cuda()
        # if torch.cuda.is_available() and cuda_device > -1:
        #     self.generator = self.generator.to(cuda_device)
        print("OK")

    def perform_generation(self, tokenizer, cuda_device=0, n_to_generate=1):
        self.generator.eval()

        for (step, inputs) in enumerate(self.dataloader):

            # retrieve the instance involved
            instance_guid = inputs["guid"].numpy()[0]
            instance_to_update = self.dataset.__getitem__(instance_guid)

            # we limit the output length to be reasonably equal to the input
            # context, i.e. the example
            max_length_example = len(tokenizer.encode(instance_to_update.text_a))
            max_length_output = int(2 * max_length_example)

            generation_arguments = {
                "max_length": max_length_output,
                "min_length": 5,
                "no_repeat_ngram_size": self.gen_cfgs["no_repeat_ngram_size"],
                "num_beams": self.gen_cfgs["num_beams"],
                "repetition_penalty": float(self.gen_cfgs["repetition_penalty"]),
                "temperature": float(self.gen_cfgs["temperature"]),
                "do_sample": False,
                "num_return_sequences": n_to_generate,
                "top_k": 10,
                "top_p": 0,
            }

            try:
                # if torch.cuda.is_available() and cuda_device > -1:
                #     inputs = inputs.to(cuda_device)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                _, generated_counter = self.generator.generate(inputs,
                                                               verbose=False,
                                                               **generation_arguments)

                # insert the generated counterfactual
                instance_to_update.meta["generated_counter"] = generated_counter[0]
                # print(generated_counter)

            except Exception as e:
                instance_to_update.meta["generated_counter"] = None
                print(f"Instance that generated the exception:{instance_guid}")
                print(e)

            if (step % 100) == 0 and (step > 0):
                print(f"{datetime.datetime.now()}, Step:{step}: 100 counterfactuals generated")

    def dataframe_from_dataset(self):
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
             "generated_counter": generated_counters
             }
        return pd.DataFrame(data=d)

    # TODO
    def perform_single_generation(self):
        pass

    # # TODO modify
    # def print_dataset(self, file_to_print, args):
    #     """Print the dataset"""
    #     df_to_print = self.dataframe_from_dataset()
    #
    #     # print such dataframe
    #     filename = f"{file_to_print[:-5]}-{args}.gen"
    #     df_to_print.to_csv(filename, sep='\t', index=False)
