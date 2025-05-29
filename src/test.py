from config import WANDB_API_KEY
import wandb
import yaml

wandb.login(key=WANDB_API_KEY)


def main():
    wandb.init(project="sweep-test")
    cfg = wandb.config
    print(cfg)


if __name__ == "__main__":
    print("Sweep test:")
    with open("tokenizer_sweep.yml", "r") as f:
        sweep_cfg = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project="sweep-test")

    wandb.agent(sweep_id, function=main)
