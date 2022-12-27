import scipy.stats as stats
import wandb
import argparse


def mcnemar(args):
    # McNemar’s test is a type of chi-square test. However, unlike the Chi-square test which is used to test
    # the relationship between 2 variables (χ2 test of Independence),
    # McNemar’s test is used to check if there are any changes in perception, attitude, behavior
    # on 2 dependent populations. The McNemar test is used whenever the same individuals are measured twice
    # (before and after survey), matched pairs (twins or married couples), matching control
    # (i.e. matched on some variable (pair of asthma patients by severity of disease in above example).

    # retrieve accuracy data from wand runs
    api = wandb.Api()
    origin_run = api.run(f"{args.wandb_project}{args.origin_id}")
    cat_run = api.run(f"{args.wandb_project}{args.cat_id}")

    n_test = args.n_test
    origin_acc = origin_run.summary["accuracy"]
    cat_acc = cat_run.summary["accuracy"]
    n_origin = int(origin_acc*n_test)
    n_cat = int(cat_acc*n_test)

    test_value = (abs(n_cat - n_origin)-1)**2 / (n_cat + n_origin)
    pvalue = 1 - stats.chi2.cdf(test_value, 1)

    if pvalue < args.alpha:
        print("The performance is significant")
    else:
        print("The performance is NOT significant")
    print(f"pvalue:{pvalue}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        required=True,
        help="The project path in wandb."
    )

    parser.add_argument(
        "--test_name",
        default=None,
        type=str,
        required=True,
        help="The test to run."
    )

    parser.add_argument(
        "--n_test",
        default=None,
        type=int,
        required=True,
        help="The # of instances in the test set."
    )

    parser.add_argument(
        "--alpha",
        default=None,
        type=float,
        required=True,
        help="The level of significance of the test."
    )

    parser.add_argument(
        "--origin_id",
        default=None,
        type=str,
        required=True,
        help="The id of the run of origin tuning."
    )

    parser.add_argument(
        "--cat_id",
        default=None,
        type=str,
        required=True,
        help="The id of the run of cat tuning."
    )

    args = parser.parse_args()

    # here
    if args.test_name == "mcnemar":
        mcnemar(args)


if __name__ == "__main__":
    main()
