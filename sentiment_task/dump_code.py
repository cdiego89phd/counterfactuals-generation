import polyjuice
from typing import Tuple
from polyjuice.polyjuice import pol

if __name__ == "__main__":
    print("ciao")

    # the base sentence
    text = "It is great for kids."
    # text = ("It is great for kids.", 1)

    pj = polyjuice.Polyjuice(model_path="uw-hai/polyjuice", is_cuda=True)
    perturbations = pj.perturb(text)


