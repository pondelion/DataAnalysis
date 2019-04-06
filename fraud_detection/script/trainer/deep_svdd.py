from typing import List
from functools import reduce
import torch
import torch.nn as nn
import numpy as np
from ..models.deep_svdd import DeepSVDD
from .. import DEVICE


def fit_deep_svdd(
    dflist: List[np.ndarray],
    input_dim: int=31,
    in_channel: int=1,
    batch_n: int=100,
    train_n: int=10000,
    learning_rate: float=1e-3,
    weight_decay: float=1e-5,
    sampling_num: int=10000,
    l2_reg_coef: float=1.0
):
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)

    model = DeepSVDD(
        in_dim=input_dim,
        in_channel=in_channel
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    try:
        from tqdm import tqdm
        itr = tqdm(range(train_n))
    except Exception:
        itr = range(train_n)

    loss_history = []

    for _ in itr:
        rd_indices = np.random.randint(0, merged.shape[0], batch_n)
        x = [torch.from_numpy(merged[idx, :]).float() for idx in rd_indices]
        x = torch.stack(x).view(batch_n, 1, merged.shape[1]).to(DEVICE)

        x_reduced = model(x)
        random_indices = np.random.permutation(merged.shape[0])[:sampling_num]
        x_all_reduced = model(torch.from_numpy(merged[random_indices, :]).float().to(DEVICE).view(sampling_num, 1, input_dim))

        # Calculate centroid in latent space
        c = torch.mean(x_all_reduced, dim=0).to(DEVICE)  # (sampling_num, 2) => (1, 2)

        loss = torch.sum((x_reduced - c)**2).to(DEVICE)

        l2_reg = None
        for w in model.parameters():
            if l2_reg is None:
                l2_reg = l2_reg_coef * w.norm(2).to(DEVICE)
            else:
                l2_reg += l2_reg_coef * w.norm(2).to(DEVICE)

        loss += l2_reg

        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history, model


def svdd_transform(
    dflist: List[np.ndarray],
    model,
):
    return [model(torch.from_numpy(df).float().view(df.shape[0], 1, df.shape[1]).to(DEVICE)).detach() for df in dflist]


def get_centroid(
    reduced_list: List[np.ndarray],
):
    return torch.stack([torch.mean(reduced, dim=0) for reduced in reduced_list]).detach()


def svdd_loss(
    dflist: List[np.ndarray],
    model,
    centroid=None,
):
    reduced_list = svdd_transform(dflist, model)
    if centroid is None:
        merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
        centroid = get_centroid([merged])[0]
    return [((reduced - centroid)**2).sum(dim=1).detach().numpy() for reduced in reduced_list]