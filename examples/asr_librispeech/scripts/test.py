import wandb

# Initialize a new wandb run
wandb.init(project="test-project")

# Log a sample metric
wandb.log({"test_metric": 1})

# Finish the run
wandb.finish()