import torch
import pandas as pd
import datetime
import pickle
import os
import time

from src.optim.algorithms import (
    StochasticSubgradientMethod,
    StochasticRegularizedDualAveraging,
    LSVRG,
    SLSVRG,
)
from src.utils.data import load_dataset
from src.utils.io import save_results, load_results, var_to_str, get_path
from src.optim.objective import (
    ORMObjective,
    get_extremile_weights,
    get_superquantile_weights,
    get_esrm_weights,
    get_erm_weights,
)

SUCCESS_CODE = 0
FAIL_CODE = -1


class OptimizationError(Exception):
    pass


def train_model(optimizer, val_objective, n_epochs, profile):
    epoch = 0
    epoch_len = optimizer.get_epoch_len()
    metrics = [compute_metrics(epoch, optimizer, val_objective, 0.0)]
    init_loss = metrics[0]["train_loss"]

    if profile:
        biases = torch.zeros(n_epochs, dtype=torch.float64)
        variances = torch.zeros(n_epochs, dtype=torch.float64)
        mses = torch.zeros(n_epochs, dtype=torch.float64)
        argsorts = torch.zeros(n_epochs, optimizer.objective.n, dtype=torch.float64)

    while epoch < n_epochs:

        tic = time.time()
        optimizer.start_epoch()
        for _ in range(epoch_len):
            optimizer.step()
        optimizer.end_epoch()
        toc = time.time()
        if profile:
            (
                biases[epoch],
                variances[epoch],
                mses[epoch],
                argsorts[epoch],
            ) = optimizer.get_profile()
        epoch += 1

        # Logging.
        metrics.append(compute_metrics(epoch, optimizer, val_objective, toc - tic))
        if metrics[-1]["train_loss"] >= 1.5 * init_loss:
            raise OptimizationError(
                f"train loss 50% greater than inital loss! (epoch {epoch})"
            )

    result = {
        "weights": optimizer.weights,
        "metrics": pd.DataFrame(metrics),
    }
    if hasattr(optimizer, "nb_checkpoints"):
        result["nb_checkpoints"] = optimizer.nb_checkpoints
    if profile:
        prof_result = {
            "biases": biases,
            "variances": variances,
            "mses": mses,
            "argsorts": argsorts,
        }
        return result, prof_result
    return result


def get_optimizer(optim_cfg, objective, seed):
    name, lr, epoch_len = (
        optim_cfg["optimizer"],
        optim_cfg["lr"],
        optim_cfg["epoch_len"],
    )
    if name == "sgd":
        return StochasticSubgradientMethod(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    elif name == "srda":
        return StochasticRegularizedDualAveraging(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    elif name == "lsvrg":
        return LSVRG(objective, lr=lr, seed=seed, epoch_len=epoch_len)
    elif name == "lsvrg_uniform":
        return LSVRG(objective, lr=lr, uniform=True, seed=seed, epoch_len=epoch_len)
    elif name == "lsvrg_uniform":
        LSVRG(objective, lr=lr, uniform=True, seed=seed, epoch_len=epoch_len)
    elif name == "slsvrg_uniform":
        return SLSVRG(objective, lr=lr, uniform=True, seed=seed, length_epoch=epoch_len)
    elif name == "slsvrg_l2":
        return SLSVRG(
            objective,
            lr=lr,
            smooth_coef=1e-3,
            smoothing="l2",
            random_checkpoint="never",
            seed=seed,
            length_epoch=epoch_len,
        )
    elif name == "slsvrg_neg_ent":
        return SLSVRG(
            objective,
            lr=lr,
            smooth_coef=1e-3,
            smoothing="neg_entropy",
            random_checkpoint="never",
            seed=seed,
            length_epoch=epoch_len,
        )
    elif name == "slsvrg_l2_rnd_check":
        return SLSVRG(
            objective,
            lr=lr,
            smooth_coef=1e-3,
            smoothing="l2",
            nb_passes=2,
            seed=seed,
            length_epoch=epoch_len,
        )
    elif name == "slsvrg_neg_ent_rnd_check":
        return SLSVRG(
            objective,
            lr=lr,
            smooth_coef=1e-3,
            smoothing="neg_entropy",
            nb_passes=2,
            seed=seed,
            length_epoch=epoch_len,
        )
    else:
        raise ValueError("Unreocgnized optimizer!")


def get_objective(model_cfg, X, y):
    name, l2_reg, loss, n_class = (
        model_cfg["objective"],
        model_cfg["l2_reg"],
        model_cfg["loss"],
        model_cfg["n_class"],
    )
    if name == "erm":
        weight_function = lambda n: get_erm_weights(n)
    elif name == "extremile":
        weight_function = lambda n: get_extremile_weights(n, 2.0)
    elif name == "superquantile":
        weight_function = lambda n: get_superquantile_weights(n, 0.5)
    elif name == "esrm":
        weight_function = lambda n: get_esrm_weights(n, 1.0)
    elif name == "extremile_lite":
        weight_function = lambda n: get_extremile_weights(n, 1.5)
    elif name == "superquantile_lite":
        weight_function = lambda n: get_superquantile_weights(n, 0.25)
    elif name == "esrm_lite":
        weight_function = lambda n: get_esrm_weights(n, 0.5)
    elif name == "extremile_hard":
        weight_function = lambda n: get_extremile_weights(n, 2.5)
    elif name == "superquantile_hard":
        weight_function = lambda n: get_superquantile_weights(n, 0.75)
    elif name == "esrm_hard":
        weight_function = lambda n: get_esrm_weights(n, 2.0)

    return ORMObjective(
        X, y, weight_function, l2_reg=l2_reg, loss=loss, n_class=n_class
    )


def compute_metrics(epoch, optimizer, val_objective, elapsed):
    return {
        "epoch": epoch,
        "train_loss": optimizer.objective.get_batch_loss(optimizer.weights).item(),
        "train_loss_unreg": optimizer.objective.get_batch_loss(
            optimizer.weights, include_reg=False
        ).item(),
        "val_loss": val_objective.get_batch_loss(optimizer.weights).item(),
        "elapsed": elapsed,
    }


def compute_training_curve(
    dataset, model_cfg, optim_cfg, seed, n_epochs, profile=False
):
    X_train, y_train, X_val, y_val = load_dataset(dataset)

    if model_cfg["loss"] == "multinomial_cross_entropy":
        model_cfg["n_class"] = len(torch.unique(y_train))
    train_objective = get_objective(model_cfg, X_train, y_train)
    val_objective = get_objective(model_cfg, X_val, y_val)
    optimizer = get_optimizer(optim_cfg, train_objective, seed)
    try:
        result = train_model(optimizer, val_objective, n_epochs, profile)
        exit_code = SUCCESS_CODE
    except OptimizationError as e:
        result = FAIL_CODE
        exit_code = FAIL_CODE
    if profile:
        save_results(result[0], dataset, model_cfg, optim_cfg, seed)
        path = get_path([dataset, var_to_str(model_cfg), optim_cfg["optimizer"]])
        f = os.path.join(path, "profile_seed_{seed}.p")
        pickle.dump(result[1], open(f, "wb"))
        return exit_code
    save_results(result, dataset, model_cfg, optim_cfg, seed)
    return exit_code


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def compute_average_train_loss(
    dataset, model_cfg, optim_cfg, seeds, out_path="results/"
):
    total = 0.0
    for seed in seeds:
        results = load_results(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            return [torch.inf]
        total += torch.tensor(results["metrics"]["train_loss"])
    return total / len(seeds)


def find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds):

    # Compute optimal hyperparameters by lowest average final train loss.
    best_loss = torch.inf
    best_traj = None
    best_cfg = None
    for optim_cfg in optim_cfgs:
        avg_train_loss = compute_average_train_loss(
            dataset, model_cfg, optim_cfg, seeds
        )
        if len(avg_train_loss) > 1 and torch.trapezoid(avg_train_loss) < best_loss:
            best_loss = torch.trapezoid(avg_train_loss)
            best_traj = avg_train_loss
            best_cfg = optim_cfg

    # Collect results for best configuration.
    df = pd.DataFrame(
        {
            "epoch": [i for i in range(len(best_traj))],
            "average_train_loss": [val.item() for val in best_traj],
        }
    )

    path = get_path([dataset, var_to_str(model_cfg), optim_cfgs[0]["optimizer"]])

    for seed in seeds:
        results = load_results(dataset, model_cfg, best_cfg, seed)
        df[f"seed_{seed}_train"] = results["metrics"]["train_loss"]
        df[f"seed_{seed}_val"] = results["metrics"]["val_loss"]
        if "nb_checkpoints" in results.keys():
            nb_checkpoints = results["nb_checkpoints"]
            pickle.dump(
                nb_checkpoints, open(os.path.join(path, "nb_checkpoints.p"), "wb")
            )
        if seed == 1:
            weights = results["weights"]

    pickle.dump(best_cfg, open(os.path.join(path, "best_cfg.p"), "wb"))
    pickle.dump(weights, open(os.path.join(path, "best_weights.p"), "wb"))
    pickle.dump(df, open(os.path.join(path, "best_traj.p"), "wb"))
