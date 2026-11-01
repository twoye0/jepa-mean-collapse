import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp02_jepa_collapse import Encoder, Predictor


torch.set_num_threads(1)


def generate_data(n=2048):
    x = torch.randn(n, 1)
    y = x**2 - 1
    return x, y


def mean_variance(z):
    return z.var(dim=0, unbiased=False).mean().item()


def evaluate(
    context_encoder,
    target_encoder,
    predictor,
    test_x,
    test_y,
):
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    with torch.no_grad():
        z_x = context_encoder(test_x)
        z_y = target_encoder(test_y)
        prediction = predictor(z_x)

        prediction_mse = F.mse_loss(
            prediction, z_y
        ).item()

        # Mean baseline uses the test-set mean here.
        # This is a descriptive reference; a train-set
        # mean should be used for a strict held-out baseline.
        mean_prediction = z_y.mean(
            dim=0, keepdim=True
        ).expand_as(z_y)

        mean_mse = F.mse_loss(
            mean_prediction, z_y
        ).item()

        prediction_variance = mean_variance(
            prediction
        )

        target_variance = mean_variance(z_y)

        improvement = (
            1 - prediction_mse / mean_mse
            if mean_mse > 1e-12
            else float("nan")
        )

    return {
        "prediction_mse": prediction_mse,
        "mean_mse": mean_mse,
        "prediction_variance": prediction_variance,
        "target_variance": target_variance,
        "improvement": improvement,
    }


def run_jepa(mode, seed):
    torch.manual_seed(seed)

    train_x, train_y = generate_data()
    test_x, test_y = generate_data()

    context_encoder = Encoder()
    target_encoder = copy.deepcopy(
        context_encoder
    )
    predictor = Predictor()

    for p in target_encoder.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(context_encoder.parameters())
        + list(predictor.parameters()),
        lr=1e-3,
    )

    tau = 0.99

    for step in range(1000):
        context_encoder.train()
        predictor.train()

        z_x = context_encoder(train_x)

        with torch.no_grad():
            z_y = target_encoder(train_y)

        prediction = predictor(z_x)

        loss = F.mse_loss(prediction, z_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if mode == "ema":
            with torch.no_grad():
                for target_param, context_param in zip(
                    target_encoder.parameters(),
                    context_encoder.parameters(),
                ):
                    target_param.mul_(tau).add_(
                        context_param,
                        alpha=1 - tau,
                    )

    result = evaluate(
        context_encoder,
        target_encoder,
        predictor,
        test_x,
        test_y,
    )

    return result


class DirectMLP(nn.Module):
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


def run_direct_mlp(seed):
    torch.manual_seed(seed)

    train_x, train_y = generate_data()
    test_x, test_y = generate_data()

    model = DirectMLP()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for step in range(1000):
        prediction = model(train_x)
        loss = F.mse_loss(prediction, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        prediction = model(test_x)

        mse = F.mse_loss(
            prediction, test_y
        ).item()

        baseline = F.mse_loss(
            test_y.mean().expand_as(test_y),
            test_y,
        ).item()

    return mse, baseline


if __name__ == "__main__":
    seeds = [42, 123, 456]

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"SEED = {seed}")
        print(f"{'=' * 60}")

        mlp_mse, mlp_baseline = run_direct_mlp(seed)

        print("\n[Direct MLP]")
        print(f"Prediction MSE : {mlp_mse:.6f}")
        print(f"Mean Baseline  : {mlp_baseline:.6f}")

        for mode in ["frozen", "ema"]:
            result = run_jepa(mode, seed)

            print(f"\n[Toy JEPA — {mode}]")
            print(
                f"Prediction MSE     : "
                f"{result['prediction_mse']:.6f}"
            )
            print(
                f"Mean Baseline MSE  : "
                f"{result['mean_mse']:.6f}"
            )
            print(
                f"Prediction Variance: "
                f"{result['prediction_variance']:.6f}"
            )
            print(
                f"Target Variance    : "
                f"{result['target_variance']:.6f}"
            )
            print(
                f"Relative Improvement: "
                f"{result['improvement']:.4f}"
            )