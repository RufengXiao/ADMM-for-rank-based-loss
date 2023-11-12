import os
import pickle
import torch
import numpy as np

from src.utils.training import compute_average_train_loss
from src.utils.io import var_to_str, get_path, load_results
from src.utils.config import N_EPOCHS


def get_suboptimality(dataset, model_cfg, train_loss, eps=1e-9, out_path="../results/"):
    init_loss = train_loss[0]

    path = get_path([dataset, var_to_str(model_cfg)], out_path=out_path)
    f = os.path.join(path, "lbfgs_min_loss.p")
    min_loss = pickle.load(open(f, "rb"))
    subopt = torch.tensor(
        (train_loss - min_loss + eps) / (init_loss - min_loss + 10 * eps)
    )
    # return torch.log10(subopt) # use if not setting yscale to log.
    return subopt


def plot_traj(
    ax, dataset, model_cfg, plot_cfg, seeds, plot_all=True, out_path="../results/"
):
    filename = plot_cfg["optimizer"]  # "code" name (e.g. "lsvrg")
    label = plot_cfg["label"]  # display name
    color = plot_cfg["color"]
    linestyle = plot_cfg["linestyle"]

    path = get_path([dataset, var_to_str(model_cfg), filename], out_path=out_path)
    if model_cfg["objective"] in plot_cfg:
        optim_cfg = {
            "optimizer": plot_cfg["optimizer"],
            "lr": plot_cfg[model_cfg["objective"]]["lr"],
            "epoch_len": plot_cfg[model_cfg["objective"]]["epoch_len"],
        }
        avg_train_loss = compute_average_train_loss(
            dataset, model_cfg, optim_cfg, seeds, out_path=out_path
        )
    else:
        df = pickle.load(open(os.path.join(path, "best_traj.p"), "rb"))
        avg_train_loss = df["average_train_loss"]
    epochs = torch.arange(len(avg_train_loss))
    subopt = get_suboptimality(dataset, model_cfg, avg_train_loss, out_path=out_path)
    if "lsvrg" in filename:
        if os.path.exists(os.path.join(path, "nb_checkpoints.p")):
            nb_checkpoints = pickle.load(
                open(os.path.join(path, "nb_checkpoints.p"), "rb")
            )
            avg_nb_passes_per_epoch = (N_EPOCHS + nb_checkpoints) / N_EPOCHS
        else:
            if "iwildcam" in dataset:
                avg_nb_passes_per_epoch = 3
            else:
                avg_nb_passes_per_epoch = 2
        x = avg_nb_passes_per_epoch * torch.arange(len(subopt))
        idx = x < len(subopt)
        ax.plot(
            x[idx], subopt[idx], color=color, label=label, linestyle=linestyle,
        )
    else:
        ax.plot(epochs, subopt, color=color, label=label, linestyle=linestyle)
    if plot_all:
        for seed in seeds:
            train_loss = df[f"seed_{seed}_train"]
            subopt = get_suboptimality(dataset, model_cfg, train_loss)
            if "lsvrg" in filename:
                if os.path.exists(os.path.join(path, "nb_checkpoints.p")):
                    nb_checkpoints = pickle.load(
                        open(os.path.join(path, "nb_checkpoints.p"), "rb")
                    )
                    avg_nb_passes_per_epoch = (N_EPOCHS + nb_checkpoints) / N_EPOCHS
                else:
                    avg_nb_passes_per_epoch = 2
                x = avg_nb_passes_per_epoch * torch.arange(len(subopt))
                idx = x < len(subopt)
                ax.plot(
                    x[idx], subopt[idx], color=color, linewidth=1, linestyle=linestyle,
                )
            else:
                ax.plot(
                    epochs,
                    subopt,
                    color=color,
                    linewidth=1,
                    linestyle=linestyle,
                    alpha=0.3,
                )


def get_runtime(dataset, model_cfg, filename, seeds, out_path):
    path = get_path([dataset, var_to_str(model_cfg), filename], out_path=out_path)
    best_cfg = pickle.load(open(os.path.join(path, "best_cfg.p"), "rb"))
    elapsed = 0.0
    for seed in seeds:
        results = load_results(dataset, model_cfg, best_cfg, seed, out_path=out_path)
        elapsed += results["metrics"]["elapsed"].to_numpy()
    return np.cumsum(elapsed / len(seeds))


def plot_runtime(ax, dataset, model_cfg, plot_cfg, seeds, out_path="../results/"):
    filename = plot_cfg["optimizer"]  # "code" name (e.g. "osvrg")
    label = plot_cfg["label"]  # display name
    color = plot_cfg["color"]
    linestyle = plot_cfg["linestyle"]
    marker = plot_cfg["marker"]

    path = get_path([dataset, var_to_str(model_cfg), filename], out_path=out_path)
    df = pickle.load(open(os.path.join(path, "best_traj.p"), "rb"))
    avg_train_loss = df["average_train_loss"]
    x = get_runtime(dataset, model_cfg, filename, seeds, out_path)
    subopt = get_suboptimality(dataset, model_cfg, avg_train_loss, out_path=out_path)
    ax.plot(x[1:], subopt[1:], color=color, linestyle=linestyle, label=label)
