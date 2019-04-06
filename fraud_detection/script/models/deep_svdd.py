import torch
import torch.nn as nn


class DeepSVDD(nn.Module):
    """
    References:
        http://data.bit.uni-bonn.de/publications/ICML2018.pdf
    """

    def __init__(self, in_dim, in_channel, base_channel=64, k_size=2, stride=2):

        super(DeepSVDD, self).__init__()

        self._in_dim = in_dim
        self._in_channel = in_channel
        self._base_channel = base_channel
        self._k_size = k_size
        self._stride = stride

        self._feat_layers = nn.ModuleDict({
            'layer1': nn.Sequential(
                nn.Conv1d(self._in_channel, self._base_channel, self._k_size, self._stride),
                nn.BatchNorm1d(self._base_channel),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            'layer2': nn.Sequential(
                nn.Conv1d(self._base_channel, 2*self._base_channel, self._k_size, self._stride),
                nn.BatchNorm1d(2*self._base_channel),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            'layer3': nn.Sequential(
                nn.Conv1d(2*self._base_channel, 4*self._base_channel, self._k_size, self._stride),
                nn.BatchNorm1d(4*self._base_channel),
                nn.LeakyReLU(negative_slope=0.1)
            ),
            'layer4': nn.Sequential(
                nn.Conv1d(4*self._base_channel, 8*self._base_channel, self._k_size, self._stride),
                nn.BatchNorm1d(8*self._base_channel),
                nn.LeakyReLU(negative_slope=0.1)
            )
        })

        classifier_in_dim = self._get_classifier_in_dim()

        self._classification_layers = nn.Sequential(
            nn.Linear(classifier_in_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        for layer in self._feat_layers.values():
            x = layer(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        x = self._classification_layers(x)
        return x

    def _get_classifier_in_dim(self):
        dummy_data = torch.randn(3, self._in_channel, self._in_dim)
        for layer in self._feat_layers.values():
            dummy_data = layer(dummy_data)
        dummy_data = dummy_data.view(dummy_data.shape[0], -1)
        return dummy_data.shape[1]
