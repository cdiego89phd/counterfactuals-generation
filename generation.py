import torch
import datetime
import openprompt
import utils
from openprompt.prompts import ManualTemplate
import traceback


class CounterGenerator:
    def __init__(self,
                 template: ManualTemplate,
                 lm,
                 dataloader: openprompt.PromptDataLoader,
                 dataset: utils.TaskDataset,
                 cfgs: dict):
        """Constructor of the counterfactual generator
        @param: template The openprompt template for the generation
        @param lm The language model used for the generation
        @param: dataloader That store the dataset
        @param: dataset A Dataset to use to perform generation
        @param: cfgs The parameters of the generation
        """
        super(CounterGenerator, self).__init__()
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

    def get_dataset(self) -> utils.TaskDataset:
        return self.dataset

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
