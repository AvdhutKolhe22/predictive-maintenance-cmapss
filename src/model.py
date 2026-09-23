import torch
import torch.nn as nn


class CNNBiLSTMMultiTask(nn.Module):
    """
    CNN + BiLSTM multi-task model for NASA C-MAPSS FD001.

    Input:
        (batch, sequence_length, num_features)

    Outputs:
        rul:
            predicted Remaining Useful Life

        failure_probability:
            probability of failure within the
            predefined prediction horizon
    """

    def __init__(
        self,
        num_features=16,
        cnn_filters=64,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.2
    ):

        super().__init__()

        # -------------------------------------------------
        # CNN feature extractor
        # -------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=num_features,
                out_channels=cnn_filters,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(
                cnn_filters
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Conv1d(
                in_channels=cnn_filters,
                out_channels=cnn_filters,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(
                cnn_filters
            ),

            nn.ReLU()
        )

        # -------------------------------------------------
        # BiLSTM temporal model
        # -------------------------------------------------

        self.bilstm = nn.LSTM(

            input_size=cnn_filters,

            hidden_size=lstm_hidden,

            num_layers=lstm_layers,

            batch_first=True,

            bidirectional=True,

            dropout=(
                dropout
                if lstm_layers > 1
                else 0.0
            )
        )

        # -------------------------------------------------
        # Shared representation
        # -------------------------------------------------

        shared_size = (
            lstm_hidden * 2
        )

        self.shared_dropout = nn.Dropout(
            dropout
        )

        # -------------------------------------------------
        # RUL regression head
        # -------------------------------------------------

        self.rul_head = nn.Sequential(

            nn.Linear(
                shared_size,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1
            )
        )

        # -------------------------------------------------
        # Failure classification head
        # -------------------------------------------------

        self.failure_head = nn.Sequential(

            nn.Linear(
                shared_size,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        # -------------------------------------------------
        # Input
        #
        # x:
        # (batch, sequence, features)
        # -------------------------------------------------

        # Conv1D expects:
        # (batch, channels, sequence)

        x = x.transpose(
            1,
            2
        )

        # -------------------------------------------------
        # CNN
        # -------------------------------------------------

        x = self.cnn(x)

        # CNN output:
        # (batch, filters, sequence)

        # Convert back for LSTM:
        # (batch, sequence, filters)

        x = x.transpose(
            1,
            2
        )

        # -------------------------------------------------
        # BiLSTM
        # -------------------------------------------------

        lstm_output, _ = self.bilstm(x)

        # -------------------------------------------------
        # Use final temporal representation
        #
        # Since this is bidirectional:
        # final representation contains both
        # forward and backward temporal information.
        # -------------------------------------------------

        representation = (
            lstm_output[:, -1, :]
        )

        representation = (
            self.shared_dropout(
                representation
            )
        )

        # -------------------------------------------------
        # Multi-task heads
        # -------------------------------------------------

        rul = self.rul_head(
            representation
        ).squeeze(-1)

        failure_logits = (
            self.failure_head(
                representation
            ).squeeze(-1)
        )

        return {
            "rul": rul,
            "failure_logits": failure_logits
        }


# ---------------------------------------------------------
# Quick architecture test
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = CNNBiLSTMMultiTask().to(
        device
    )

    dummy_input = torch.randn(
        8,
        30,
        16,
        device=device
    )

    outputs = model(
        dummy_input
    )

    print(
        "Device:",
        device
    )

    print(
        "Input shape:",
        dummy_input.shape
    )

    print(
        "RUL output:",
        outputs["rul"].shape
    )

    print(
        "Failure logits:",
        outputs["failure_logits"].shape
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )