from torch import Tensor, nn

from vad.modeling.transformer import SinusoidalPositionalEncoding, TransformerEncoder


class SelfAttentiveVAD(nn.Module):
    def __init__(self, feature_size: int, num_layers: int, d_model: int, dropout: float):
        super(SelfAttentiveVAD, self).__init__()

        d_ff = d_model * 4

        self.input_layer = nn.Sequential(
            nn.Linear(feature_size, d_model),
            SinusoidalPositionalEncoding(encoding_size=d_model, initial_length=10),
            nn.Dropout(dropout),
        )
        self.encoder = TransformerEncoder(
            num_layers=num_layers, d_model=d_model, d_ff=d_ff, n_heads=1, dropout=dropout
        )
        self.classifier = nn.Linear(d_model, 2)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, features: Tensor):
        x = self.input_layer(features)
        x = self.encoder(x)
        x = self.classifier(x)
        x = self.log_softmax(x)
        return x
