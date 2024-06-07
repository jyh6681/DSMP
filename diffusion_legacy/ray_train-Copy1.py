import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import sMP2

class ExperimentTerminationReporter(CLIReporter):
    def should_report(self, trials, done=True):
        """Reports only on experiment termination."""
        return done
    
def training_function(config):
    # Hyperparameters
    # modes, width, learning_rate = config["modes"], config["width"], config["learning_rate"]
#     for step in range(10):
    # Iterative training function - can be any arbitrary training procedure.
    # intermediate_score = fourier_2d_hopt_train.objective(modes, width, learning_rate)
    # Feed the score back back to Tune.
    acc = sMP2.objective(**config)
    tune.report(acc=acc)
        
ray.shutdown()
ray.init(num_cpus=8, num_gpus=2) 

asha_scheduler = ASHAScheduler(
    # time_attr='training_iteration',
    metric='loss',
    mode='min',
    max_t=400,
    grace_period=100,
    reduction_factor=2,
    brackets=1)


analysis = tune.run(
    training_function,
    config={
            
            "learning_rate": tune.grid_search([0.01, 0.02, 0.05, 0.005]),
            "weight_decay": tune.grid_search([1e-3, 1e-2, 2e-2]),

            # "GN": tune.grid_search(False),
            "nhid" : tune.grid_search( [16, 24, 32, 48, 64, 86]),
            "dropout_prob": tune.grid_search([0.5, 0.6, 0.4, 0.2, 0.7]), 
            # "tqdm_disable": tune.grid_search(True),
        
            },
    progress_reporter=ExperimentTerminationReporter(),
    resources_per_trial={'gpu': 1, 'cpu': 4},
    # scheduler=asha_scheduler
    )

print("Best config: ", analysis.get_best_config(
    metric="acc", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df



        