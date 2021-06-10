import torch
from torch import Tensor, nn


# Naming of modules follows the paper
class ACAM(nn.Module):
    def __init__(
        self,
        window_feature_size,
        window_size: int,
        core_hidden_size: int,
        encoder_hidden_size: int,
        encoder_output_size: int,
        dropout: float,
        num_steps: int,
    ):
        super(ACAM, self).__init__()

        self.input_dropout = nn.Dropout(p=dropout)
        self.decoder = Decoder(core_hidden_size=core_hidden_size, window_size=window_size)
        self.attention = Attention()
        self.encoder = Encoder(
            window_feature_size=window_feature_size,
            window_size=window_size,
            encoder_hidden_size=encoder_hidden_size,
            encoder_output_size=encoder_output_size,
        )
        self.core = Core(encoder_output_size, core_hidden_size, dropout, False)
        self.classifier = Classifier(core_hidden_size, 7)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.num_steps = num_steps

    def forward(self, features: Tensor):
        # features: (batch_size, window_size, feature_size)

        batch_size, window_size, feature_size = features.size()
        features = self.input_dropout(features)

        initial_attention = (
            torch.ones(batch_size, window_size, dtype=torch.float32, device=features.device)
            / window_size
        )
        attended_input = self.attention(selected_input=features, attention=initial_attention)
        aggregation = self.encoder(attention=initial_attention, attended_input=attended_input)
        core_output, core_state = self.core(aggregation=aggregation)

        for _ in range(self.num_steps):
            attention = self.decoder(core_output)
            attended_input = self.attention(selected_input=features, attention=attention)
            aggregation = self.encoder(attention=attention, attended_input=attended_input)
            core_output, core_state = self.core(aggregation=aggregation, state=core_state)

        classifier_output = self.classifier(core_output=core_output).view(
            batch_size, window_size, 2
        )
        return self.log_softmax(classifier_output)


class Decoder(nn.Module):
    def __init__(self, core_hidden_size: int, window_size: int):
        super(Decoder, self).__init__()

        self.transform = nn.Sequential(
            nn.Linear(core_hidden_size, window_size), nn.BatchNorm1d(num_features=window_size)
        )
        self.smooth_softmax = SmoothSoftmax()

    def forward(self, core_output: Tensor):
        # core_output: (batch_size, core_hidden_size)
        attention = self.smooth_softmax(self.transform(core_output))
        return attention


class Attention(nn.Module):
    def forward(self, selected_input: Tensor, attention: Tensor):
        # selected_input: (batch_size, window_size, feature_size)
        # attention: (batch_size, window_size)

        attended_input = selected_input * attention.unsqueeze(-1)
        return attended_input


class Encoder(nn.Module):
    def __init__(
        self,
        window_feature_size: int,
        window_size: int,
        encoder_hidden_size: int,
        encoder_output_size: int,
    ):
        super(Encoder, self).__init__()

        self.transform_attention = nn.Sequential(
            nn.Linear(window_size, encoder_hidden_size),
            nn.BatchNorm1d(num_features=encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, encoder_output_size),
            nn.BatchNorm1d(num_features=encoder_output_size),
        )
        self.transform_attended_input = nn.Sequential(
            nn.Linear(window_feature_size, encoder_hidden_size),
            nn.BatchNorm1d(num_features=encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, encoder_output_size),
            nn.BatchNorm1d(num_features=encoder_output_size),
        )
        self.aggregate = nn.ReLU()

    def forward(self, attention: Tensor, attended_input: Tensor):
        batch_size, window_size, feature_size = attended_input.size()
        attended_input = attended_input.view(batch_size, window_size * feature_size)
        return self.aggregate(
            self.transform_attention(attention) + self.transform_attended_input(attended_input)
        )


class Core(nn.Module):
    def __init__(
        self, encoder_output_size: int, hidden_size: int, dropout: float, bidirectional: bool
    ):
        super(Core, self).__init__()

        self.lstm = nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.recurrent_dropout = nn.Dropout(p=dropout)

    def forward(self, aggregation: Tensor, state=None):
        aggregation = aggregation.unsqueeze(dim=1)  # Make sequence length of 1
        if state is not None:
            # This implementation of layer norm & dropout for rnn could be different from that of the paper
            state = [self.recurrent_dropout(self.layer_norm(s)) for s in state]
        core_output, core_state = self.lstm(aggregation, state)
        core_output = core_output.squeeze(dim=1)
        return core_output, core_state


class Classifier(nn.Module):
    def __init__(self, core_hidden_size: int, window_size: int):
        super(Classifier, self).__init__()

        self.transform = nn.Linear(
            core_hidden_size, window_size * 2
        )  # Different from the paper's one dimension output

    def forward(self, core_output: Tensor):
        return self.transform(core_output)


class SmoothSoftmax(nn.Module):
    def forward(self, x: Tensor):
        logistic_value = torch.sigmoid(x)
        return logistic_value / logistic_value.sum(dim=-1, keepdim=True)
