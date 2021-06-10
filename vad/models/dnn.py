from torch import Tensor, nn


class DNN(nn.Module):
    def __init__(
        self,
        window_feature_size: int,
        first_hidden_features: int,
        second_hidden_features: int,
        dropout: float,
    ):
        super(DNN, self).__init__()

        self.dnn = nn.Sequential(
            nn.Dropout(p=dropout),  # not in reference code
            nn.Linear(window_feature_size, first_hidden_features),
            nn.BatchNorm1d(first_hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(first_hidden_features, second_hidden_features),
            nn.BatchNorm1d(second_hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(second_hidden_features, 2),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, features: Tensor):
        # features: (batch_size, window_size, feature_size)

        x = features.view(features.size(0), -1)
        x = self.dnn(x)
        x = self.log_softmax(x)

        return x
