import wandb
import random


def train():
    # wandb.init(group="experiment_1")
    wandb.init()
    # print(f"Agent that runs training with hypers:")
    # print(wandb.config)
    loss = random.random()
    wandb.log(dict(loss=loss))
    print(f"Loss reported:{loss}")
    wandb.finish()


def main():

    wandb.login()
    # wandb.init(project="counterfactual-generation")

    # sweep_id = wandb.sweep(project="pytorch-sweeps-demo")
    sweep_id = "cdiego89/counterfactual-generation/k9btxxrf"
    print(f"Sweep id:{sweep_id}")
    # sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    wandb.agent(sweep_id, train)

    # sweep_url = sweep_run.get_sweep_url()
    # project_url = sweep_run.get_project_url()
    # sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    #
    # print(f"Sweep url:{sweep_url}")
    # print(f"Group url:{sweep_group_url}")
    # print(f"Notes:{sweep_run.notes}")

    # sweep_run.save()
    # sweep_run_name = sweep_run.name or sweep_run.id or "unknown"


if __name__ == "__main__":
    main()
