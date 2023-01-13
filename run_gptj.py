from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch


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

    if args.precision == 16:
        precision = torch.float16
    else:
        precision = torch.float32

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",
                                                 torch_dtype=precision)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)


if __name__ == "__main__":
    main()
