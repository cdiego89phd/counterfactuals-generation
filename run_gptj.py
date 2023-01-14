from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import utils


PRECISION_DICT = {16: torch.float16,
                  32: torch.float32
                  }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        required=True,
        help=""
    )
    args = parser.parse_args()
    precision = PRECISION_DICT[args.precision]

    utils.print_gpu_utilization()
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",
                                                 torch_dtype=precision)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    prompt = (
        "In a shocking finding, scientists discovered "
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids.cuda(),
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
    print()
    utils.print_gpu_utilization()


if __name__ == "__main__":
    main()
