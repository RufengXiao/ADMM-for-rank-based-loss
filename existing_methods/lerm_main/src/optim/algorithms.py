import torch
import numpy as np
from existing_methods.lerm_main.src.utils.smoothing import get_smooth_weights


class Optimizer:
    def __init__(self):
        pass

    def start_epoch(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_epoch(self):
        raise NotImplementedError

    def get_epoch_len(self):
        raise NotImplementedError


class SubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01):
        super(SubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )

    def start_epoch(self):
        pass

    def step(self):
        g = self.objective.get_batch_subgrad(self.weights)
        self.weights = self.weights - self.lr * g

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return 1


class StochasticSubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01, batch_size=64, seed=25, epoch_len=None):
        super(StochasticSubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr
        self.batch_size = batch_size

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
        self.order = None
        self.iter = None
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0

    def step(self):
        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        self.weights.requires_grad = True
        g = self.objective.get_batch_subgrad(self.weights, idx=idx)
        self.weights.requires_grad = False
        self.weights = self.weights - self.lr * g
        self.iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class StochasticRegularizedDualAveraging(Optimizer):
    def __init__(
        self, objective, lr=0.01, l2_reg=1.0, batch_size=64, seed=25, epoch_len=None
    ):
        super(StochasticRegularizedDualAveraging, self).__init__()
        self.objective = objective
        self.aux_reg = 1 / lr
        self.l2_reg = l2_reg
        self.batch_size = batch_size

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
            self.dual_avg = torch.zeros(
                objective.n_class * self.objective.d, dtype=torch.float64
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
            self.dual_avg = torch.zeros(self.objective.d, dtype=torch.float64)

        self.order = None
        self.epoch_iter = None
        self.total_iter = 0
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.epoch_iter = 0

    def step(self):
        idx = self.order[
            self.epoch_iter
            * self.batch_size : min(
                self.objective.n, (self.epoch_iter + 1) * self.batch_size
            )
        ]
        g = self.objective.get_batch_subgrad(self.weights, idx=idx, include_reg=False)
        self.dual_avg = (self.total_iter * self.dual_avg + g) / (self.total_iter + 1)
        self.weights = -self.dual_avg / (
            self.l2_reg / self.objective.n + self.aux_reg / (self.total_iter + 1)
        )
        self.weights.requires_grad = True
        self.epoch_iter += 1
        self.total_iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class LSVRG(Optimizer):
    def __init__(self, objective, lr=0.01, uniform=False, seed=25, epoch_len=None):
        super(LSVRG, self).__init__()
        self.objective = objective
        self.lr = lr
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
        self.alphas = self.objective.alphas
        if self.objective.weight_function2 is not None:
            self.betas = self.objective.betas
            self.lossB = torch.tensor(self.objective.lossB)
        else:
            self.betas = None
            self.lossB = None
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = self.objective.n

    def start_epoch(self):
        losses = self.objective.get_indiv_loss(self.weights, with_grad=True)
        sort, argsort = torch.sort(losses, stable=True)
        if self.objective.weight_function2 is not None:
            risk = torch.dot(self.alphas[sort <= self.lossB], sort[sort <= self.lossB])
            risk += torch.dot(self.betas[sort > self.lossB], sort[sort > self.lossB])
        else:
            risk = torch.dot(self.alphas, sort)

        self.subgrad_checkpt = torch.autograd.grad(outputs=risk, inputs=self.weights)[0]
        self.argsort_checkpt = argsort
        self.weights_checkpt = torch.clone(self.weights)

    def step(self):
        n = self.objective.n

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
            x = self.objective.X[i]
            y = self.objective.y[i]
        else:
            i = torch.tensor([np.random.choice(n, p=self.alphas)])
            x = self.objective.X[self.argsort_checkpt[i]]
            y = self.objective.y[self.argsort_checkpt[i]]

        # Compute gradient at current iterate.
        loss_cur = self.objective.loss(self.weights, x, y)
        g = torch.autograd.grad(outputs=loss_cur, inputs=self.weights)[0]

        # Compute gradient at previous checkpoint.
        loss = self.objective.loss(self.weights_checkpt, x, y)
        g_checkpt = torch.autograd.grad(outputs=loss, inputs=self.weights_checkpt)[0]

        if self.uniform:
            if self.lossB is not None:
                if loss_cur <= self.lossB:
                    direction = n * self.alphas[i] * (g - g_checkpt) + self.subgrad_checkpt
                else:
                    direction = n * self.betas[i] * (g - g_checkpt) + self.subgrad_checkpt
            else:
                direction = n * self.alphas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction += self.objective.l2_reg * self.weights / n
        if self.objective.l1_reg:
            # l1 = torch.abs(self.weights * self.objective.l1_reg / (2 * n)).sum()
            # direction += torch.autograd.grad(l1, self.weights)[0]
            res = torch.zeros(self.weights.shape)
            res[self.weights > 0] = 1
            res[self.weights < 0] = -1
            res[self.weights == 0] = torch.rand(1) * 2 - 1
            direction += res * self.objective.l1_reg / (2 * n)

        self.weights = self.weights - self.lr * direction

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class SLSVRG(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=1.0,
        smoothing="l2",
        random_checkpoint="1/n",
        seed=25,
        length_epoch=None,
    ):
        super(SLSVRG, self).__init__()
        n = objective.n
        self.objective = objective
        self.lr = lr
        # adjust for multiclass classification
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
        self.spectrum = self.objective.alphas
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        if random_checkpoint == "never":
            self.proba_checkpoint = 0
        elif random_checkpoint == "1/n":
            self.proba_checkpoint = 1 / n
        else:
            raise NotImplementedError
        self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smoothing = smoothing
        if length_epoch:
            self.length_epoch = length_epoch
        else:
            self.length_epoch = int(nb_passes * n)
        self.nb_checkpoints = 0

    def start_epoch(self):
        losses = self.objective.get_indiv_loss(self.weights, with_grad=True)
        with torch.no_grad():
            self.alphas = get_smooth_weights(
                losses, self.spectrum, self.smooth_coef, self.smoothing
            )
        risk = torch.dot(self.alphas, losses)

        self.subgrad_checkpt = torch.autograd.grad(outputs=risk, inputs=self.weights)[0]
        self.weights_checkpt = torch.clone(self.weights)
        self.nb_checkpoints += 1

    def step(self):
        n = self.objective.n

        q = torch.rand(1)
        if q <= self.proba_checkpoint:
            losses = self.objective.get_indiv_loss(self.weights, with_grad=True)
            risk = torch.dot(self.alphas, losses)
            self.subgrad_checkpt = torch.autograd.grad(
                outputs=risk, inputs=self.weights
            )[0]
            self.weights_checkpt = torch.clone(self.weights)
            self.nb_checkpoints += 1

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
        else:
            i = torch.tensor([np.random.choice(n, p=self.alphas)])
        x = self.objective.X[i]
        y = self.objective.y[i]

        # Compute gradient at current iterate.
        loss = self.objective.loss(self.weights, x, y)
        g = torch.autograd.grad(outputs=loss, inputs=self.weights)[0]

        # Compute gradient at previous checkpoint.
        loss = self.objective.loss(self.weights_checkpt, x, y)
        g_checkpt = torch.autograd.grad(outputs=loss, inputs=self.weights_checkpt)[0]

        if self.uniform:
            direction = n * self.alphas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction += self.objective.l2_reg * self.weights / n

        self.weights = self.weights - self.lr * direction

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch
