import torch
import torch.nn as nn

torch.manual_seed(42)


# 1. Dataset
def generate_data(rho, n=10000):
    x = torch.randn(n, 1)
    noise = torch.randn(n, 1)

    y = rho * x + (1 - rho**2) ** 0.5 * noise

    return x, y


# 2. Predictor
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# 3. Experiment
def run_experiment(rho):
    x_train, y_train = generate_data(rho)
    x_test, y_test = generate_data(rho)

    model = Predictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(2000):
        prediction = model(x_train)
        loss = ((prediction - y_train) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = model(x_test)

        mse = ((prediction - y_test) ** 2).mean().item()
        prediction_variance = prediction.var(unbiased=False).item()

        # Use the training-set mean for the baseline.
        mean_prediction = y_train.mean()
        mean_baseline_mse = (
            (mean_prediction - y_test) ** 2
        ).mean().item()

    return mse, prediction_variance, mean_baseline_mse


# 4. Run
if __name__ == "__main__":
    for rho in [0.0, 0.5, 1.0]:
        mse, variance, baseline = run_experiment(rho)

        print(f"\nrho = {rho}")
        print(f"Prediction MSE      : {mse:.4f}")
        print(f"Prediction Variance : {variance:.4f}")
        print(f"Mean Baseline MSE   : {baseline:.4f}")