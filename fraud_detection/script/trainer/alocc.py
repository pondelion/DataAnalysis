from typing import List
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from ..models.auto_encoder import AutoEncoder, CNNAutoEncoder
from ..models.alocc import Discriminator
from .. import DEVICE


def fit_alocc(
    dflist: List[np.ndarray],
    input_dim: int,
    batch_n: int=100,
    train_n: int=20000,
    learning_rate: float=1e-3,
    output_dim: int=2,
    ae_type: str='cnn'
):
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
    data_n = merged.shape[0]

    if ae_type == 'fc':
        netR = AutoEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        ).to(DEVICE)
    elif ae_type == 'cnn':
        netR = CNNAutoEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        ).to(DEVICE)
    netD = Discriminator(in_dim=input_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizerR = torch.optim.Adam(
        netR.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    optimizerD = torch.optim.Adam(
        netD.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    loss_history = []

    try:
        from tqdm import tqdm
        itr = tqdm(range(train_n))
    except Exception:
        itr = range(train_n)

    print('Train start.')
    for n in itr:
        rd_indices = np.random.permutation(data_n)[:batch_n]
        minibatch = merged[rd_indices, :]
        if ae_type == 'fc':
            minibatch = torch.stack([torch.from_numpy(minibatch[i, :]).float() for i in range(minibatch.shape[0])]).to(DEVICE)
        elif ae_type == 'cnn':
            minibatch = torch.stack([torch.from_numpy(minibatch[i, :]).float().unsqueeze(0) for i in range(minibatch.shape[0])]).to(DEVICE)

        generated = netR(minibatch + 0.1*minibatch.max()*torch.randn(minibatch.shape).to(DEVICE))

        # Discriminator Network
        netD.zero_grad()

        label = torch.full((batch_n, 1), 1, device=DEVICE)
        output = netD(minibatch)
        errD_real = criterion(output, label)
        errD_real.backward()

        fake = netR(generated)
        label.fill_(0)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()

        optimizerD.step()

        # Reinforcer Network
        netR.zero_grad()

        label.fill_(1)
        output = netD(fake)
        errR = 10*criterion(output, label)
        errR.backward()
        optimizerR.step()

        loss_history.append([
            errD_real.data[0],
            errD_fake.data[0],
            errR.data[0]
        ])

    return loss_history, netD, netR
