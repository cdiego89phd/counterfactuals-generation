import wandb


def train():
    with wandb.init():
        config_dict = wandb.config
        # for key in config_dict:
        #     print(f"{key}:{config_dict[key]}")

        print("#########################################################")
        print("#########################################################")
        print("")
        wandb.log({"eval/loss": 1})


def main():

    n_sweep_runs = 2
    wandb_key = "ac0eb1b13268d81f2526a3d354e135e6a1ede08c"
    wandb.login(relogin=True, key=wandb_key)
    sweep_id = f"cdiego89/counterfactual-generation/reo8x3sd"
    print(f"Sweep id:{sweep_id}")

    wandb.agent(sweep_id, function=train, count=n_sweep_runs)


if __name__ == "__main__":
    main()
