import torch
import torch.nn as nn


class GaitModel(nn.Module):
    def __init__(self, num_joints, num_people):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_joints * 2,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.embedding = nn.Linear(256, 128)

        # Person identification head
        self.id_head = nn.Linear(128, num_people)

        # Gender classification head
        self.gender_head = nn.Linear(128, 2)

    def forward(self, x):
        """
        x shape: (B, T, J, 2)
        """
        B, T, J, C = x.shape
        x = x.view(B, T, J * C)

        _, (h, _) = self.lstm(x)
        h = h[-1]  # last layer hidden state

        embedding = self.embedding(h)

        id_logits = self.id_head(embedding)
        gender_logits = self.gender_head(embedding)



        return embedding, id_logits, gender_logits

