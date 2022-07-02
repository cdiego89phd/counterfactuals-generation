import argparse
import yaml
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import datetime
import itertools
import bs4

import transformers
import openprompt
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
from openprompt.plms.lm import LMTokenizerWrapper


def load_raw_dataset(loading_path):
    train = pd.read_csv(loading_path + "training_set", sep='\t')
    val = pd.read_csv(loading_path + "val_set", sep='\t')
    test = pd.read_csv(loading_path + "test_set", sep='\t')
    return train, val, test


def augment_dataset(df_dataset):
    booked_ids = df_dataset["paired_id"].values
    examples = df_dataset["example"].values
    labels_ex = df_dataset["label_ex"].values
    counters = df_dataset["counterfactual"].values
    labels_counters = df_dataset["label_counter"].values

    ids = generate_custom_ids(booked_ids)
    d = {"paired_id": ids,
         "example": counters,
         "label_ex": labels_counters,
         "counterfactual": examples,
         "label_counter": labels_ex}
    new_df = pd.DataFrame(data=d)

    # append the new df
    df_final = pd.concat([df_dataset, new_df], ignore_index=True)

    return df_final


def generate_custom_ids(idxs):
    max_id = max(idxs)
    return [i for i in range(max_id + 1, max_id + 1 + len(idxs))]


def wrap_with_prompt(df_row, template, special_tokens, map_labels):
    final_text = template.replace("<label_ex>", map_labels[df_row["label_ex"]])
    final_text = final_text.replace("<example_text>", df_row["example"])
    final_text = final_text.replace("<label_counter>", map_labels[df_row["label_counter"]])

    if special_tokens is not None and "sep_token" in special_tokens:
        final_text = final_text.replace("<sep>", special_tokens["sep_token"])
    if special_tokens is not None and "bos_token" in special_tokens:
        final_text = final_text.replace("<bos_token>", special_tokens["bos_token"])
    return final_text


class SentimentDataset(Dataset):
    def __init__(self, raw_dataframe):
        # get a copy of the dataframe
        self.raw_dataframe = raw_dataframe.copy(deep=True)
        self.guids = []
        self.dataset = {}

    # convert the Dataframe into the InputExample format dataset of openprompt
    def prepare_dataloader(self):
        for index, row in self.raw_dataframe.iterrows():
            self.dataset[row['paired_id']] = InputExample(guid=row['paired_id'],
                                                          text_a=bs4.BeautifulSoup(
                                                              row['wrapped_input'], "lxml").text,
                                                          meta={"label_ex":row['label_ex'],
                                                                "label_counter":row['label_counter'],
                                                                'example':bs4.BeautifulSoup(
                                                                    row['example'], "lxml").text,
                                                                'counterfactual':bs4.BeautifulSoup(
                                                                    row['counterfactual'], "lxml").text})
            self.guids.append(row['paired_id'])
        print('Dataloader prepared!')

    # the same of __getitem__
    def get_instance_by_id(self, idx):
        return self.dataset[idx]

    # implemented because of inheritance from Dataset
    def __len__(self):
        return len(self.dataset)

    # implemented because of inheritance from Dataset
    def __iter__(self):
        return iter(self.dataset)

    def __next__(self):
        return iter(self.dataset)

    # implemented because of inheritance from Dataset
    def __getitem__(self, idx):
        return self.dataframe.__getitem__(idx)

    def get_dataset(self):
        return self.dataset

    def get_raw_dataframe(self):
        return self.raw_dataframe


def set_generator(template, plm, parallelization, cuda_gen):

    prompt = openprompt.PromptForGeneration(
        template = template,
        freeze_plm = True,
        plm = plm,
        plm_eval_mode = True
    )

    if torch.cuda.is_available() and cuda_gen:
        prompt = prompt.cuda()

    if parallelization:
        prompt.parallelize()

    return prompt


class CounterGenerator:
    def __init__(self, dataloader, dataset, generator, tok, cfgs):
        self.dataloader = dataloader
        self.dataset = dataset
        self.generator = generator
        self.tok = tok
        self.gen_cfgs = cfgs

    def perform_generation(self, on_cuda):
        self.generator.eval()
        if torch.cuda.is_available() and on_cuda:
            print(f"Total GPU memory available: {torch.cuda.get_device_properties(0).total_memory}")
            print(f"Allocated GPU memory before generation: {torch.cuda.memory_allocated(0)}")
            print(f"Allocated GPU memory reserved: {torch.cuda.memory_reserved(0)}")

        for (step, inputs) in enumerate(self.dataloader):

            # retrieve the instance involved
            instance_guid = inputs["guid"].numpy()[0]
            instance_to_update = self.dataset.get_instance_by_id(instance_guid)

            # we limit the output length to be reasonably equal to the input
            # context, i.e. the example
            max_length_example = len(self.tok.encode(instance_to_update.text_a))
            max_length_output = int(2 * max_length_example)

            # cfg_gen[0] = no_repeat_ngram_size
            # cfg_gen[1] = num_beam
            # cfg_gen[2] = repetition_penalty
            # cfg_gen[3] = temperature
            generation_arguments = {
                "max_length": max_length_output,
                "min_length": 5,
                "no_repeat_ngram_size": self.gen_cfgs[0],
                "num_beams": self.gen_cfgs[1],
                "repetition_penalty": self.gen_cfgs[2],
                "temperature": self.gen_cfgs[3],
                "do_sample": False,
                "top_k": 10,
                "top_p": 0,
            }

            try:
                if torch.cuda.is_available() and on_cuda:
                    inputs = inputs.cuda()
                _, generated_counter = self.generator.generate(inputs,
                                                               verbose=False,
                                                               **generation_arguments)

                # insert the generated counterfactual
                instance_to_update.meta["generated_counter"] = generated_counter[0]
                # print(generated_counter)

            except Exception as e:
                instance_to_update.meta["generated_counter"] = None
                print(instance_guid)
                print(e)

            if (step % 100) == 0 and (step > 0):
                print(f"{datetime.datetime.now()}, Step:{step}: 100 counterfactuals generated")

    def dataframe_from_dataset(self):
        paired_ids = [idx for idx in self.dataset]
        labels_ex = [self.dataset.get_instance_by_id(idx).meta["label_ex"] for idx in self.dataset]
        examples = [self.dataset.get_instance_by_id(idx).meta["example"] for idx in self.dataset]
        labels_counter = [self.dataset.get_instance_by_id(idx).meta["label_counter"] for idx in self.dataset]
        counterfactuals = [self.dataset.get_instance_by_id(idx).meta["counterfactual"] for idx in self.dataset]
        generated_counters = [self.dataset.get_instance_by_id(idx).meta["generated_counter"] for idx in self.dataset]
        d = {"paired_id": paired_ids,
             "label_ex": labels_ex,
             "example": examples,
             "label_counter": labels_counter,
             "counterfactual": counterfactuals,
             "generated_counter": generated_counters
             }
        return pd.DataFrame(data=d)

    def print_generation(self, path_to_print, args):
        # create a dataframe from dataset
        df_to_print = self.dataframe_from_dataset()

        # print such dataframe
        filename = f"{path_to_print[:-5]}-{args}.gen"
        df_to_print.to_csv(filename, sep='\t', index=False)


def load_language_model_objects(model_name, special_tokens, sst2_model_path):
    if model_name == "gpt2-fine-tuned-sst2":  # load the sst2 fine tuned model
        load_path = f"{sst2_model_path}{model_name}"
    else:  # load model from the hugging face repository
        load_path = model_name

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(load_path)
    print("Downloaded tokenizer!")
    if special_tokens is not None:
        print(f"Len of tokenizer before adding tokens:{len(tokenizer)}")
        tokenizer.add_special_tokens(special_tokens)  # add special tokens
        print("Added special tokens to tokenizer!")
        print(f"Len of tokenizer after adding tokens:{len(tokenizer)}")

    # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    lm_config_class = transformers.GPT2Config.from_pretrained(load_path, pad_token_id=tokenizer.eos_token_id)

    lm = transformers.GPT2LMHeadModel.from_pretrained(load_path, config=lm_config_class)
    if special_tokens is not None:
        # Special tokens added, model needs to be resized accordingly
        lm.resize_token_embeddings(len(tokenizer))

    # load lm class for the tokenizer (for the generation with openprompt)
    tokenizer_wrapper = LMTokenizerWrapper

    print("Downloaded tokenizer, model and cfg!")

    return tokenizer, tokenizer_wrapper, lm, lm_config_class


def main():
    # load required args from command line
    parser = argparse.ArgumentParser()

    # SETTING_PATH = "/home/diego/counterfactuals-generation/sentiment_task/zs_gpt2_experiments/settings/"
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

    args = parser.parse_args()

    setting_yaml_file = open(f"{args.setting_path}{args.setting_name}")
    parsed_yaml_file = yaml.load(setting_yaml_file, Loader=yaml.FullLoader)

    # the following params will be included in a yaml file
    RESULTS_PATH = parsed_yaml_file['RESULTS_PATH']
    SST2_MODEL_PATH = parsed_yaml_file['SST2_MODEL_PATH']
    RANDOM_SEED_SHUFFLE = parsed_yaml_file['RANDOM_SEED_SHUFFLE']
    AUGMENT_VALSET = parsed_yaml_file['AUGMENT_VALSET']
    KEEP_FIRST_N = parsed_yaml_file['KEEP_FIRST_N']
    FOLDS = parsed_yaml_file['FOLDS']
    MODEL_NAME = parsed_yaml_file['MODEL_NAME']
    SPECIAL_TOKENS = parsed_yaml_file['SPECIAL_TOKENS']
    TEMPLATE_PROMPT = parsed_yaml_file['TEMPLATE_PROMPT']
    MAP_LABELS = parsed_yaml_file['MAP_LABELS']
    ON_CUDA = parsed_yaml_file['ON_CUDA']
    PARALLELIZATION = parsed_yaml_file['PARALLELIZATION']
    GEN_ARGS = parsed_yaml_file['GEN_ARGS']
    print("Experiment's params read from yaml file")

    tokenizer, tokenizer_wrapper, lm, lm_config_class = load_language_model_objects(MODEL_NAME,
                                                                                    SPECIAL_TOKENS,
                                                                                    SST2_MODEL_PATH)

    template_prompt = '{"placeholder":"text_a"}{"mask"}'
    prompt_template = ManualTemplate(text=template_prompt, tokenizer=tokenizer)

    all_pars = sorted(GEN_ARGS)
    gen_grid = list(itertools.product(*(GEN_ARGS[par] for par in all_pars)))

    for fold in FOLDS:

        # create dir to store the results
        res_path = f"{RESULTS_PATH}fold_{fold}"
        if not os.path.exists(res_path):
            # Create a new directory because it does not exist
            os.makedirs(res_path)

        print(f"{datetime.datetime.now()}: Beginning generation for fold {fold}")
        for gen_args in gen_grid:
            print(f"\nGeneration parameters: {gen_args}")
            # load the datasets
            df_trainset, df_valset, df_testset = load_raw_dataset(f"../cad_imdb/fold_{fold}/")

            # shuffle valset
            df_valset = df_valset.sample(frac=1, random_state=RANDOM_SEED_SHUFFLE)

            # whether to duplicate the data by inverting example-counter the intances
            print(f"# of instances in the validation set:{len(df_valset)}")
            if AUGMENT_VALSET:
                df_valset = augment_dataset(df_valset)
                print(f"Augmented dataset - # of instances in the validation set:{len(df_valset)}")

            # whether to reduce the valset
            if KEEP_FIRST_N > 0:
                df_valset = df_valset.head(KEEP_FIRST_N)

            # wrap the datasets with the prompt template
            df_valset["wrapped_input"] = df_valset.apply(lambda row: wrap_with_prompt(row,
                                                                                      TEMPLATE_PROMPT,
                                                                                      SPECIAL_TOKENS,
                                                                                      MAP_LABELS), axis=1)

            # prepare the data loader
            valset = SentimentDataset(raw_dataframe=df_valset)
            valset.prepare_dataloader()

            val_data_loader = openprompt.PromptDataLoader(
                dataset=list(valset.get_dataset().values()),
                tokenizer=tokenizer,
                template=prompt_template,
                tokenizer_wrapper_class=tokenizer_wrapper
            )

            # set the prompt for generation
            prompt_for_generation = set_generator(prompt_template,
                                                  lm,
                                                  PARALLELIZATION,
                                                  ON_CUDA)

            # generate counterfactuals
            counter_generator = CounterGenerator(val_data_loader,
                                                 valset,
                                                 prompt_for_generation,
                                                 tokenizer,
                                                 gen_args
                                                 )
            counter_generator.perform_generation(ON_CUDA)

            # print the generated counterfactuals
            print(f"{datetime.datetime.now()}: Printing generation...")
            counter_generator.print_generation(f"{RESULTS_PATH}fold_{fold}/{args.setting_name}", gen_args)
            print(f"{datetime.datetime.now()}: Finished to print...")

        print(f"{datetime.datetime.now()}: Generation completed for fold {fold}")


if __name__ == "__main__":
    main()
