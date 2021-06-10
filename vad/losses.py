from torch import nn


class TokenNLLLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()

        self.nll_loss = nn.NLLLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, outputs, targets):
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)

        loss = self.nll_loss(outputs_flat, targets_flat)
        return loss
