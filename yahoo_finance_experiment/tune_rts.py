import os
import sys
import numpy as np
import optuna

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# We import the exact same data downloader and engine
from yahoo_finance_experiment.yfdataset import download_stock_data
from yahoo_finance_experiment.rts_portfolio.rts_engine import ReactiveTabuSearch

print("Loading data for tuning...")
stock_names, returns_array = download_stock_data()
cov_matrix = np.cov(returns_array, rowvar=False)
n_assets = len(stock_names)
print("Data loaded.")

def objective(trial):
    """
    Optuna objective function to maximize the Sharpe Ratio
    by dynamically suggesting different RTS hyperparameters.
    """
    neighbors_size = trial.suggest_int("neighbors_size", 10, 200)
    initial_tenure = trial.suggest_int("initial_tenure", 5, 50)
    beta = trial.suggest_float("beta", 1.0, 2.0)
    osc_period = trial.suggest_int("osc_period", 10, 100)
    
    # We use a smaller max_iter for faster tuning trials
    # In production, you would bump this back to 5000
    max_iter = 1000
    n_runs = 3  # Multiple runs to handle randomness
    
    sharpes = []
    for _ in range(n_runs):
        engine = ReactiveTabuSearch(
            returns_data=returns_array,
            cov_matrix=cov_matrix,
            n_assets=n_assets,
            max_iter=max_iter,
            neighbors_size=neighbors_size,
            initial_tenure=initial_tenure,
            beta=beta,
            osc_period=osc_period,
        )
        res = engine.run()
        sharpes.append(res['best_sharpe'])
        
    return np.mean(sharpes)

if __name__ == "__main__":
    print("\n=======================================================")
    print("  RTS HYPERPARAMETER TUNING (Bayesian Optimization)")
    print("=======================================================")
    
    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    
    # Let's run a quick 15 trials test
    study.optimize(objective, n_trials=15)
    
    print("\n[✓] Tuning Complete.")
    print(f"Best Trial Average Sharpe Ratio: {study.best_value:.4f}")
    print("\nBest Discovered Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    print("\nYou can now plug these parameters back into run_rts_portfolio.py!")
